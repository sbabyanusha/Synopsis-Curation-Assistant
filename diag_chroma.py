import streamlit as st, sys, platform, importlib, os

st.title("🧬 Chroma diagnostics")
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())
st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))

def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"FAIL: {e}"

mods = ["numpy", "pandas", "chromadb", "langchain", "langchain_community", "langchain_openai", "pypdf"]
for m in mods:
    st.write(m, "→", ver(m))

# smoke test the sqlite patch and chroma persistent client
try:
    import sys
    import sqlite3
    import chromadb
    st.write("sqlite3 module:", sqlite3.__name__, getattr(sqlite3, "sqlite_version", "n/a"))
    client = chromadb.PersistentClient(path="./chroma_index")
    # create + drop a collection
    col = client.get_or_create_collection("diag_collection")
    col.add(ids=["a"], documents=["hello world"])
    st.success("Chroma PersistentClient OK (DuckDB+Parquet)")
except Exception as e:
    st.exception(e)
