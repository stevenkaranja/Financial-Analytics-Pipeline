import streamlit as st
import sqlite3
import pandas as pd
import os
from pathlib import Path
import io

# Path to the main database
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
DB_PATH = project_root / "data" / "database" / "finance_data.db"

def main():
    st.set_page_config(page_title="DB Explorer", page_icon="🗄️", layout="wide")
    st.title("🗄️ Database Explorer")
    db_path = str(DB_PATH)
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}")
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Table introspection
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
    st.sidebar.header("Tables")
    selected_table = st.sidebar.selectbox("Select a table", tables)
    if selected_table:
        cursor.execute(f"PRAGMA table_info({selected_table})")
        schema = cursor.fetchall()
        st.write(f"**Schema for `{selected_table}`:**")
        st.table(pd.DataFrame(schema, columns=["cid", "name", "type", "notnull", "default", "pk"]))
    st.markdown("---")
    st.subheader("Run SQL Query")
    default_query = f"SELECT * FROM {selected_table} LIMIT 100" if selected_table else ""
    sql_query = st.text_area("SQL Query", value=default_query, height=100)
    if st.button("Execute Query"):
        try:
            df = pd.read_sql_query(sql_query, conn)
            st.dataframe(df)
            st.success(f"Returned {len(df)} rows.")
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, f"query_results.csv", "text/csv")
            # Excel download (optional, only if openpyxl is installed)
            try:
                import openpyxl
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                st.download_button("Download Excel", output.getvalue(), f"query_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:
                pass
        except Exception as e:
            st.error(f"Query failed: {e}")
    st.markdown("---")
    st.subheader("Upload Data to Table")
    upload_table = st.selectbox("Target table for upload", tables, key="upload_table")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file and upload_table:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)
            df_up.to_sql(upload_table, conn, if_exists='append', index=False)
            st.success(f"Uploaded {len(df_up)} rows to {upload_table}.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
    conn.close()

if __name__ == "__main__":
    main()
