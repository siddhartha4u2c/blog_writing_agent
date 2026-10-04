import io
import json
import os
import re
import sys
import types
import zipfile
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = APP_DIR / "blog_research_writing_agent_with_image.ipynb"

try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR / ".env")
except ImportError:
    pass

if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

st.set_page_config(page_title="Blog Studio", page_icon="✍️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip().startswith("run("):
            continue
        if source.strip():
            code_cells.append(source)

    module = types.ModuleType("blog_agent_runtime")
    module.__file__ = str(NOTEBOOK_PATH)
    sys.modules[module.__name__] = module
    source = "\n\n".join(code_cells)
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), module.__dict__)
    return module.__dict__


def initial_state(topic):
    return {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }


def safe_title(title):
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", title).strip(" .")
    return title or "Generated Blog"


def make_image_filenames_safe(specs):
    for spec in specs:
        filename = str(spec.get("filename", "diagram.png")).replace("\\", "/")
        spec["filename"] = Path(filename).name or "diagram.png"


LOCAL_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\((images/[^)\s]+)\)")


def render_blog(markdown):
    # The browser cannot load images/<file> from the server's disk, so each
    # local image is sent through st.image and the text around it as Markdown.
    parts = LOCAL_IMAGE_PATTERN.split(markdown)
    for index, part in enumerate(parts):
        if index % 2 == 0:
            if part.strip():
                st.markdown(part)
        elif Path(part).is_file():
            st.image(part)
        else:
            st.warning(f"Image file not found: {part}")


def blog_zip(title, markdown, image_paths):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{title}.md", markdown)
        for path in image_paths:
            archive.write(path, path.as_posix())
    return buffer.getvalue()


REVISE_OUTLINE_INSTRUCTIONS = """A reviewer rejected the current outline. Revise it to apply their feedback.

Rules:
- Apply the feedback fully, even where it overrides the section-count or word-count defaults.
- Keep everything the feedback does not touch as it is.
- Number task ids from 1 in reading order.
"""

REVISE_DRAFT_SYSTEM = """You are a senior technical editor.
A reviewer rejected the current Markdown draft of a blog post. Revise it to apply their feedback.

Rules:
- Apply the feedback fully.
- Keep everything the feedback does not touch as it is, including the H1 title, code blocks and source links.
- Keep every [[IMAGE_n]] placeholder exactly as written, unless the feedback asks to remove that image.
- Do not invent sources or URLs.
- Output ONLY the full revised Markdown, with no commentary and no code fence around the whole document.
"""


def message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


def strip_outer_fence(markdown):
    match = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n(.*)\n```\s*", markdown, re.DOTALL)
    return match.group(1) if match else markdown


def revise_outline(pipeline, state, feedback):
    planner = pipeline["llm"].with_structured_output(pipeline["Plan"])
    evidence = [item.model_dump() for item in state.get("evidence", [])][:16]
    plan = planner.invoke(
        [
            pipeline["SystemMessage"](content=pipeline["ORCH_SYSTEM"]),
            pipeline["HumanMessage"](
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {state['mode']}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n{evidence}\n\n"
                    f"{REVISE_OUTLINE_INSTRUCTIONS}\n"
                    f"Current outline:\n{state['plan'].model_dump_json(indent=2)}\n\n"
                    f"Reviewer feedback:\n{feedback}"
                )
            ),
        ]
    )
    for task_id, task in enumerate(plan.tasks, start=1):
        task.id = task_id
    return plan


def revise_draft(pipeline, state, draft, feedback):
    evidence_text = "\n".join(
        f"- {item.title} | {item.url}" for item in state.get("evidence", [])[:20]
    )
    result = pipeline["llm"].invoke(
        [
            pipeline["SystemMessage"](content=REVISE_DRAFT_SYSTEM),
            pipeline["HumanMessage"](
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Audience: {state['plan'].audience}\n"
                    f"Tone: {state['plan'].tone}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n\n"
                    f"Reviewer feedback:\n{feedback}\n\n"
                    f"Current draft:\n{draft}"
                )
            ),
        ]
    )
    revised = strip_outer_fence(message_text(result)).strip()
    if not revised:
        raise RuntimeError("The model returned an empty draft.")
    return revised + "\n"


def reset_workflow():
    for key in (
        "workflow",
        "workflow_stage",
        "draft_editor",
        "final_blog",
        "outline_rejecting",
        "outline_feedback",
        "draft_rejecting",
        "draft_feedback",
    ):
        st.session_state.pop(key, None)
    st.rerun()


st.title("Blog Studio")
st.progress(
    {"topic": 0.0, "outline": 0.35, "draft": 0.7, "done": 1.0}.get(
        st.session_state.get("workflow_stage", "topic"), 0.0
    )
)

with st.sidebar:
    st.subheader("API status")
    st.write("Gemini", "Ready" if os.getenv("GOOGLE_API_KEY") else "Key missing")
    st.write("Tavily", "Ready" if os.getenv("TAVILY_API_KEY") else "Key missing")
    if st.button("Start a new blog"):
        reset_workflow()

stage = st.session_state.get("workflow_stage", "topic")

notice = st.session_state.pop("notice", None)
if notice:
    st.warning(notice)

revised_notice = st.session_state.pop("revised_notice", None)
if revised_notice:
    st.success(revised_notice)

if stage == "topic":
    topic = st.text_input("Blog topic", placeholder="For example: QKV attention in Python")
    if st.button("Create research and outline", type="primary", disabled=not topic.strip()):
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env before starting.")
        else:
            try:
                with st.spinner("Researching and creating an outline..."):
                    pipeline = load_pipeline()
                    state = initial_state(topic.strip())
                    state.update(pipeline["router_node"](state))
                    if state["needs_research"]:
                        if not os.getenv("TAVILY_API_KEY"):
                            raise RuntimeError("This topic needs research; set TAVILY_API_KEY in .env.")
                        state.update(pipeline["research_node"](state))
                    state.update(pipeline["orchestrator_node"](state))

                st.session_state.workflow = state
                st.session_state.workflow_stage = "outline"
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create the outline: {exc}")

if stage == "outline":
    state = st.session_state.workflow
    plan = state["plan"]
    st.subheader(plan.blog_title)
    st.write(f"**Audience:** {plan.audience}  ·  **Tone:** {plan.tone}  ·  **Mode:** {state['mode']}")

    if state.get("evidence"):
        with st.expander(f"Review research sources ({len(state['evidence'])})"):
            for item in state["evidence"]:
                st.markdown(f"- [{item.title or item.url}]({item.url})")

    for task in plan.tasks:
        with st.expander(f"{task.id}. {task.title}"):
            st.write(task.goal)
            st.markdown("\n".join(f"- {bullet}" for bullet in task.bullets))
            st.caption(f"Target: {task.target_words} words")

    if st.session_state.get("outline_rejecting"):
        with st.form("outline_feedback_form"):
            feedback = st.text_area(
                "What should change in the outline?",
                placeholder="For example: drop the security section and add one on benchmarking.",
                key="outline_feedback",
            )
            revise_col, discard_col, cancel_col = st.columns(3)
            revise = revise_col.form_submit_button("Revise outline with feedback", type="primary")
            discard = discard_col.form_submit_button("Discard and start over")
            cancel = cancel_col.form_submit_button("Cancel")

        if revise:
            if not feedback.strip():
                st.error("Enter feedback to revise the outline, or discard it to start over.")
            else:
                try:
                    with st.spinner("Revising the outline with your feedback..."):
                        state["plan"] = revise_outline(load_pipeline(), state, feedback.strip())
                    st.session_state.workflow = state
                    st.session_state.pop("outline_rejecting", None)
                    st.session_state.pop("outline_feedback", None)
                    st.session_state.revised_notice = "Outline revised with your feedback. Approve it or reject it again."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not revise the outline: {exc}")
        if discard:
            st.session_state.notice = f"Outline for \"{state['topic']}\" rejected. Nothing was saved."
            reset_workflow()
        if cancel:
            st.session_state.pop("outline_rejecting", None)
            st.rerun()
    else:
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve outline and continue", type="primary"):
            st.session_state.workflow_stage = "draft"
            st.rerun()
        if reject_col.button("Reject outline"):
            st.session_state.outline_rejecting = True
            st.rerun()

if stage == "draft":
    state = st.session_state.workflow
    pipeline = load_pipeline()

    if not state.get("merged_md"):
        try:
            with st.spinner(f"Writing {len(state['plan'].tasks)} sections..."):
                plan = state["plan"]
                worker_context = {
                    "topic": state["topic"],
                    "mode": state["mode"],
                    "plan": plan.model_dump(),
                    "evidence": [item.model_dump() for item in state.get("evidence", [])],
                }
                for task in plan.tasks:
                    result = pipeline["worker_node"](
                        {**worker_context, "task": task.model_dump()}
                    )
                    state["sections"].extend(result["sections"])

                state.update(pipeline["merge_content"](state))
                state.update(pipeline["decide_images"](state))
                st.session_state.workflow = state
        except Exception as exc:
            st.error(f"Could not draft the blog: {exc}")
            st.stop()

    st.subheader("Review the draft")
    if state.get("image_specs"):
        st.caption(f"Planned diagrams: {len(state['image_specs'])}")
        for spec in state["image_specs"]:
            st.write(f"**{spec.get('caption', 'Diagram')}**: {spec.get('alt', '')}")

    draft = st.text_area(
        "Edit the Markdown, then approve it to generate images and save the final blog.",
        value=state.get("md_with_placeholders") or state["merged_md"],
        height=520,
        key="draft_editor",
    )
    title = safe_title(state["plan"].blog_title)
    output_path = Path.cwd() / f"{title}.md"
    overwrite = False
    if output_path.exists():
        overwrite = st.checkbox(f"Overwrite {output_path.name}")

    if st.session_state.get("draft_rejecting"):
        with st.form("draft_feedback_form"):
            feedback = st.text_area(
                "What should change in the draft?",
                placeholder="For example: shorten the introduction and add a code example to section 3.",
                key="draft_feedback",
            )
            revise_col, discard_col, cancel_col = st.columns(3)
            revise = revise_col.form_submit_button("Revise draft with feedback", type="primary")
            discard = discard_col.form_submit_button("Discard draft and return to outline")
            cancel = cancel_col.form_submit_button("Cancel")

        if revise:
            if not feedback.strip():
                st.error("Enter feedback to revise the draft, or discard it to return to the outline.")
            elif not draft.strip():
                st.error("The draft is empty, so there is nothing to revise.")
            else:
                try:
                    with st.spinner("Revising the draft with your feedback..."):
                        revised = revise_draft(pipeline, state, draft, feedback.strip())
                    state["md_with_placeholders"] = revised
                    state["image_specs"] = [
                        spec for spec in state.get("image_specs", []) if spec["placeholder"] in revised
                    ]
                    st.session_state.workflow = state
                    st.session_state.pop("draft_editor", None)
                    st.session_state.pop("draft_rejecting", None)
                    st.session_state.pop("draft_feedback", None)
                    st.session_state.revised_notice = "Draft revised with your feedback. Approve it or reject it again."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not revise the draft: {exc}")
        if discard:
            state.update(sections=[], merged_md="", md_with_placeholders="", image_specs=[])
            st.session_state.workflow = state
            st.session_state.pop("draft_editor", None)
            st.session_state.pop("draft_rejecting", None)
            st.session_state.pop("draft_feedback", None)
            st.session_state.workflow_stage = "outline"
            st.session_state.notice = "Draft rejected. Nothing was saved. Approve the outline to write a new draft, or reject it to change it."
            st.rerun()
        if cancel:
            st.session_state.pop("draft_rejecting", None)
            st.rerun()
    else:
        approve_col, reject_col = st.columns(2)
        if reject_col.button("Reject draft"):
            st.session_state.draft_rejecting = True
            st.rerun()

        if approve_col.button(
            "Approve and write final blog",
            type="primary",
            disabled=not draft.strip() or (output_path.exists() and not overwrite),
        ):
            try:
                state["plan"] = state["plan"].model_copy(update={"blog_title": title})
                state["md_with_placeholders"] = draft
                make_image_filenames_safe(state.get("image_specs", []))
                with st.spinner("Generating approved images and saving the blog..."):
                    result = pipeline["generate_and_place_images"](state)
                st.session_state.final_blog = result["final"]
                st.session_state.workflow_stage = "done"
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save the blog: {exc}")

if stage == "done":
    state = st.session_state.workflow
    title = safe_title(state["plan"].blog_title)
    st.success(f"Saved {title}.md in {Path.cwd()}")
    final_blog = st.session_state.final_blog
    image_paths = [Path(path) for path in dict.fromkeys(LOCAL_IMAGE_PATTERN.findall(final_blog))]
    image_paths = [path for path in image_paths if path.is_file()]

    markdown_col, zip_col = st.columns(2)
    markdown_col.download_button(
        "Download Markdown",
        data=final_blog,
        file_name=f"{title}.md",
        mime="text/markdown",
    )
    if image_paths:
        zip_col.download_button(
            "Download Markdown with images (.zip)",
            data=blog_zip(title, final_blog, image_paths),
            file_name=f"{title}.zip",
            mime="application/zip",
        )
    render_blog(final_blog)