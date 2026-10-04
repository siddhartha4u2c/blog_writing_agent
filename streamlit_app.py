import json
import os
import re
import sys
import types
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


def reset_workflow():
    for key in ("workflow", "workflow_stage", "draft_editor", "final_blog"):
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

    if st.button("Approve outline and continue", type="primary"):
        st.session_state.workflow_stage = "draft"
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

    if st.button(
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
    st.download_button(
        "Download Markdown",
        data=st.session_state.final_blog,
        file_name=f"{title}.md",
        mime="text/markdown",
    )
    st.markdown(st.session_state.final_blog)