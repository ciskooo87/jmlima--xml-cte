import os
import re
import zipfile
import hashlib
import sqlite3
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st

APP_TITLE = "CT-e Ops Control — XML + Status + Financeiro"
DB_PATH = os.path.join("data", "app.db")

STATUS_VALUES = ["PENDENTE", "OPERADO", "CANCELADO"]


# =========================================================
# Utils
# =========================================================
def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def make_batch_id():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rnd = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    return f"BATCH-{ts}-{rnd}"

def normalize_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None

def to_float_any(x: str | None) -> float | None:
    if not x:
        return None
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except:
        return None

def localname(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


# =========================================================
# XML parsing (robusto)
# =========================================================
def find_text_anywhere(root: ET.Element, tag_local: str) -> str | None:
    for el in root.iter():
        if localname(el.tag) == tag_local:
            if el.text and el.text.strip():
                return el.text.strip()
    return None

def find_node_anywhere(root: ET.Element, tag_local: str) -> ET.Element | None:
    for el in root.iter():
        if localname(el.tag) == tag_local:
            return el
    return None

def find_first_from(node: ET.Element, path_locals: list[str]) -> str | None:
    nodes = [node]
    for name in path_locals:
        next_nodes = []
        for n in nodes:
            for ch in list(n):
                if localname(ch.tag) == name:
                    next_nodes.append(ch)
        nodes = next_nodes
        if not nodes:
            return None
    return normalize_text(nodes[0].text if nodes else None)

def extract_cte_key(xml_bytes: bytes) -> tuple[str, str]:
    try:
        root = ET.fromstring(xml_bytes)
        ch = find_text_anywhere(root, "chCTe")
        if ch and re.fullmatch(r"\d{44}", ch):
            return ch, "chCTe"
    except ET.ParseError:
        pass
    return sha256_hex(xml_bytes), "hash"

def extract_fields(xml_bytes: bytes) -> dict:
    out = {"emit_cnpj": None, "vTPrest": None, "dhEmi": None, "toma_uf": None}
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out

    inf = find_node_anywhere(root, "infCte") or root

    out["emit_cnpj"] = normalize_text(find_first_from(inf, ["emit", "CNPJ"]))
    out["vTPrest"] = to_float_any(find_first_from(inf, ["vPrest", "vTPrest"]))

    dh = find_first_from(inf, ["ide", "dhEmi"]) or find_first_from(inf, ["ide", "dEmi"])
    out["dhEmi"] = normalize_text(dh)

    uf = (
        find_first_from(inf, ["ide", "toma4", "UF"]) or
        find_first_from(inf, ["ide", "toma3", "toma", "UF"]) or
        find_first_from(inf, ["compl", "enderToma", "UF"]) or
        find_first_from(inf, ["dest", "enderDest", "UF"])
    )
    out["toma_uf"] = normalize_text(uf)

    return out


# =========================================================
# DB
# =========================================================
def ensure_db():
    os.makedirs("data", exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            uploaded_at_utc TEXT NOT NULL,
            operator TEXT,
            notes TEXT,
            instituicao_financeira TEXT,
            default_status TEXT
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS cte (
            cte_key TEXT PRIMARY KEY,
            first_seen_batch_id TEXT NOT NULL,
            first_seen_at_utc TEXT NOT NULL,
            last_seen_batch_id TEXT NOT NULL,
            last_seen_at_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            filename_hint TEXT,
            source_hash TEXT,

            emit_cnpj TEXT,
            vTPrest REAL,
            dhEmi TEXT,
            toma_uf TEXT,
            instituicao_financeira TEXT
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS batch_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            cte_key TEXT NOT NULL,
            filename TEXT,
            is_duplicate INTEGER NOT NULL,
            reason TEXT,
            FOREIGN KEY(batch_id) REFERENCES batches(batch_id),
            FOREIGN KEY(cte_key) REFERENCES cte(cte_key)
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS status_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cte_key TEXT NOT NULL,
            event_at_utc TEXT NOT NULL,
            operator TEXT,
            from_status TEXT,
            to_status TEXT NOT NULL,
            reason TEXT,
            related_batch_id TEXT,
            FOREIGN KEY(cte_key) REFERENCES cte(cte_key)
        )
        """)
        con.commit()

def upsert_batch(batch_id: str, operator: str | None, notes: str | None, inst: str | None, default_status: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT INTO batches(batch_id, uploaded_at_utc, operator, notes, instituicao_financeira, default_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (batch_id, utc_now_iso(), operator, notes, inst, default_status),
        )
        con.commit()


# =========================================================
# Upload handling
# =========================================================
def read_uploaded_files(uploaded_files):
    items = []
    if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".zip"):
        zf = zipfile.ZipFile(uploaded_files[0])
        for name in zf.namelist():
            if name.lower().endswith(".xml"):
                items.append({"filename": name, "xml_bytes": zf.read(name)})
        return items

    for uf in uploaded_files:
        if uf.name.lower().endswith(".xml"):
            items.append({"filename": uf.name, "xml_bytes": uf.read()})
    return items


# =========================================================
# Core processing
# =========================================================
def process_lot(batch_id: str, files: list[dict], default_status: str, instituicao_financeira: str | None):
    results = []
    now = utc_now_iso()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row

        for f in files:
            raw = (f["xml_bytes"] or b"").strip()
            if not raw:
                continue

            cte_key, method = extract_cte_key(raw)
            src_hash = sha256_hex(raw)
            filename = f["filename"]
            fields = extract_fields(raw)

            row = con.execute("SELECT * FROM cte WHERE cte_key = ?", (cte_key,)).fetchone()

            if row is None:
                con.execute(
                    """
                    INSERT INTO cte (
                        cte_key, first_seen_batch_id, first_seen_at_utc,
                        last_seen_batch_id, last_seen_at_utc,
                        status, filename_hint, source_hash,
                        emit_cnpj, vTPrest, dhEmi, toma_uf, instituicao_financeira
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cte_key, batch_id, now,
                        batch_id, now,
                        default_status, filename, src_hash,
                        fields["emit_cnpj"], fields["vTPrest"], fields["dhEmi"], fields["toma_uf"],
                        instituicao_financeira
                    ),
                )

                con.execute(
                    """
                    INSERT INTO status_events (cte_key, event_at_utc, operator, from_status, to_status, reason, related_batch_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cte_key, now, None, None, default_status, "status inicial no upload", batch_id),
                )

                con.execute(
                    """
                    INSERT INTO batch_items(batch_id, cte_key, filename, is_duplicate, reason)
                    VALUES (?, ?, ?, 0, ?)
                    """,
                    (batch_id, cte_key, filename, f"novo ({method})"),
                )

                results.append({
                    "filename": filename,
                    "cte_key": cte_key,
                    "status_lote": "NOVO",
                    "status_atual": default_status,
                    "metodo_id": method,
                    **fields,
                    "instituicao_financeira": instituicao_financeira,
                    "first_seen_batch_id": batch_id,
                    "first_seen_at_utc": now,
                })

            else:
                con.execute(
                    """
                    UPDATE cte
                    SET last_seen_batch_id = ?, last_seen_at_utc = ?
                    WHERE cte_key = ?
                    """,
                    (batch_id, now, cte_key),
                )
                con.execute(
                    """
                    INSERT INTO batch_items(batch_id, cte_key, filename, is_duplicate, reason)
                    VALUES (?, ?, ?, 1, ?)
                    """,
                    (batch_id, cte_key, filename, "já existe no histórico"),
                )

                results.append({
                    "filename": filename,
                    "cte_key": cte_key,
                    "status_lote": "DUPLICADO",
                    "status_atual": row["status"],
                    "metodo_id": method,
                    "emit_cnpj": row["emit_cnpj"],
                    "vTPrest": row["vTPrest"],
                    "dhEmi": row["dhEmi"],
                    "toma_uf": row["toma_uf"],
                    "instituicao_financeira": row["instituicao_financeira"],
                    "first_seen_batch_id": row["first_seen_batch_id"],
                    "first_seen_at_utc": row["first_seen_at_utc"],
                })

        con.commit()

    return pd.DataFrame(results)


# =========================================================
# Queries / Management
# =========================================================
def get_recent_batches(limit=15):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT batch_id, uploaded_at_utc, operator, notes, instituicao_financeira, default_status
            FROM batches
            ORDER BY uploaded_at_utc DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])

def search_ctes(filters: dict, limit: int = 500):
    where = []
    args = []

    if filters.get("status"):
        where.append("status = ?")
        args.append(filters["status"])
    if filters.get("emit_cnpj"):
        where.append("emit_cnpj = ?")
        args.append(filters["emit_cnpj"])
    if filters.get("toma_uf"):
        where.append("toma_uf = ?")
        args.append(filters["toma_uf"])
    if filters.get("instituicao_financeira"):
        where.append("instituicao_financeira = ?")
        args.append(filters["instituicao_financeira"])

    sql = "SELECT * FROM cte"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY first_seen_at_utc DESC LIMIT ?"
    args.append(limit)

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, tuple(args)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])

def get_cte(cte_key: str):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM cte WHERE cte_key = ?", (cte_key,)).fetchone()
        return dict(r) if r else None

def update_status(cte_keys: list[str], new_status: str, operator: str | None, reason: str | None):
    if not cte_keys:
        return 0
    now = utc_now_iso()

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row

        q_marks = ",".join(["?"] * len(cte_keys))
        current = con.execute(
            f"SELECT cte_key, status FROM cte WHERE cte_key IN ({q_marks})",
            tuple(cte_keys),
        ).fetchall()
        current_map = {r["cte_key"]: r["status"] for r in current}

        params = [new_status] + cte_keys
        cur = con.execute(f"UPDATE cte SET status = ? WHERE cte_key IN ({q_marks})", params)

        for k in cte_keys:
            con.execute(
                """
                INSERT INTO status_events (cte_key, event_at_utc, operator, from_status, to_status, reason, related_batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (k, now, operator, current_map.get(k), new_status, reason, None),
            )

        con.commit()
        return cur.rowcount

def update_instituicao(cte_keys: list[str], instituicao: str | None):
    if not cte_keys:
        return 0
    with sqlite3.connect(DB_PATH) as con:
        q_marks = ",".join(["?"] * len(cte_keys))
        params = [instituicao] + cte_keys
        cur = con.execute(
            f"UPDATE cte SET instituicao_financeira = ? WHERE cte_key IN ({q_marks})",
            params
        )
        con.commit()
        return cur.rowcount

def get_status_events(cte_key: str, limit: int = 50):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT event_at_utc, operator, from_status, to_status, reason, related_batch_id
            FROM status_events
            WHERE cte_key = ?
            ORDER BY event_at_utc DESC
            LIMIT ?
            """,
            (cte_key, limit),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
ensure_db()

st.title(APP_TITLE)
st.caption("Upload XML/ZIP → extração → dedupe → histórico + workflow + instituição financeira.")

with st.sidebar:
    st.subheader("Parâmetros operacionais")
    operator = st.text_input("Operador (opcional)", value="")
    notes = st.text_area("Observações (opcional)", value="", height=80)

    st.subheader("Instituição financeira")
    inst = st.text_input("Instituição (ex: Itaú, Bradesco, Santander)", value="")

    st.subheader("Status default para novos CT-e")
    default_status = st.selectbox("Status inicial", STATUS_VALUES, index=0)

    st.divider()
    st.subheader("Consulta rápida")
    key_query = st.text_input("Buscar CT-e por chave (44 dígitos ou hash)", value="")
    if st.button("Consultar", use_container_width=True):
        if key_query.strip():
            info = get_cte(key_query.strip())
            if info:
                st.success("Encontrado no histórico.")
                st.json(info)
                ev = get_status_events(key_query.strip())
                if not ev.empty:
                    st.write("Trilha de status:")
                    st.dataframe(ev, use_container_width=True, hide_index=True)
            else:
                st.warning("Não encontrado no histórico.")


colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("1) Subir lote")
    uploaded = st.file_uploader(
        "Envie múltiplos XMLs ou 1 ZIP com XMLs",
        type=["xml", "zip"],
        accept_multiple_files=True,
    )

    if st.button("Processar lote", type="primary", disabled=not uploaded):
        files = read_uploaded_files(uploaded)
        if not files:
            st.error("Nenhum XML válido encontrado. Envie .xml ou .zip com .xml.")
            st.stop()

        batch_id = make_batch_id()
        upsert_batch(
            batch_id=batch_id,
            operator=operator.strip() or None,
            notes=notes.strip() or None,
            inst=inst.strip() or None,
            default_status=default_status,
        )

        df = process_lot(
            batch_id=batch_id,
            files=files,
            default_status=default_status,
            instituicao_financeira=inst.strip() or None
        )

        st.success(f"Lote processado: {batch_id}")
        total = len(df)
        novos = int((df["status_lote"] == "NOVO").sum()) if total else 0
        dup = int((df["status_lote"] == "DUPLICADO").sum()) if total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total no lote", total)
        m2.metric("Novos (inseridos)", novos)
        m3.metric("Duplicados (bloqueados)", dup)

        st.divider()
        st.subheader("Resultado do lote")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Baixar CSV do lote",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{batch_id}_resultado.csv",
            mime="text/csv",
        )

with colB:
    st.subheader("2) Histórico recente (lotes)")
    hist = get_recent_batches(limit=15)
    if hist.empty:
        st.info("Sem lotes ainda.")
    else:
        st.dataframe(hist, use_container_width=True, hide_index=True)


st.divider()
st.subheader("3) Gestão de status (massivo)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    f_status = st.selectbox("Filtrar status", ["(todos)"] + STATUS_VALUES, index=0)
with c2:
    f_uf = st.text_input("Filtrar UF (toma)", value="")
with c3:
    f_cnpj = st.text_input("Filtrar CNPJ emitente", value="")
with c4:
    f_inst = st.text_input("Filtrar instituição", value="")

filters = {
    "status": None if f_status == "(todos)" else f_status,
    "toma_uf": f_uf.strip() or None,
    "emit_cnpj": f_cnpj.strip() or None,
    "instituicao_financeira": f_inst.strip() or None,
}

df_cte = search_ctes(filters, limit=500)
st.caption(f"Mostrando até 500 registros. Encontrados: {len(df_cte)}")

if df_cte.empty:
    st.info("Nenhum registro no filtro atual.")
else:
    show_cols = [
        "cte_key", "status", "emit_cnpj", "vTPrest", "dhEmi", "toma_uf",
        "instituicao_financeira", "first_seen_at_utc", "first_seen_batch_id"
    ]
    st.dataframe(df_cte[show_cols], use_container_width=True, hide_index=True)

    st.markdown("### Ações massivas")

    all_keys = df_cte["cte_key"].tolist()

    # --- Selecionar tudo: controla o multiselect via session_state
    if "selected_keys" not in st.session_state:
        st.session_state.selected_keys = []

    select_all = st.checkbox("Selecionar tudo (do filtro atual)", value=False)

    if select_all:
        st.session_state.selected_keys = all_keys
    else:
        # se desmarcar, não zera automaticamente (evita perda acidental)
        pass

    keys_to_change = st.multiselect(
        "CT-e selecionados",
        options=all_keys,
        key="selected_keys"
    )

    a1, a2 = st.columns([1, 2])
    with a1:
        new_status = st.selectbox("Novo status", STATUS_VALUES, index=1)  # default OPERADO
    with a2:
        reason = st.text_input("Motivo (auditoria)", value="atualização operacional")

    b1, b2 = st.columns([2, 1])
    with b1:
        new_inst = st.text_input("Nova instituição (para os selecionados)", value="")
    with b2:
        st.write("")  # spacing

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

    with col_btn1:
        if st.button("Aplicar novo status", disabled=not keys_to_change):
            changed = update_status(keys_to_change, new_status, operator.strip() or None, reason.strip() or None)
            st.success(f"Atualizados: {changed} CT-e → {new_status}.")
            st.warning(
                "Se você filtrou por status, CT-e podem sumir da tabela após a mudança (é esperado). "
                "Use '(todos)' para ver geral."
            )

    with col_btn2:
        if st.button("Aplicar instituição", disabled=not keys_to_change):
            changed = update_instituicao(keys_to_change, new_inst.strip() or None)
            st.success(f"Atualizados: {changed} CT-e com instituição '{new_inst.strip() or None}'.")

    with col_btn3:
        if st.button("Limpar seleção", disabled=not keys_to_change):
            st.session_state.selected_keys = []
            st.success("Seleção limpa.")

st.divider()
st.subheader("4) Auditoria de status (por CT-e)")
audit_key = st.text_input("Digite uma chave CT-e para ver a trilha", value="")
if audit_key.strip():
    ev = get_status_events(audit_key.strip())
    if ev.empty:
        st.info("Sem eventos de status para essa chave (ou chave não encontrada).")
    else:
        st.dataframe(ev, use_container_width=True, hide_index=True)

st.caption(
    "Nota: SQLite no Streamlit Cloud funciona para MVP. Para missão crítica/concorrência, Postgres externo é o caminho certo."
)
