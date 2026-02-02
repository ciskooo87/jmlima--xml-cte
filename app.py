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


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def make_batch_id():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rnd = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    return f"BATCH-{ts}-{rnd}"

def to_float_br(x: str | None) -> float | None:
    if not x:
        return None
    # vTPrest normalmente vem com ponto. Mas vamos tolerar vírgula.
    s = x.strip().replace(".", ".").replace(",", ".")
    try:
        return float(s)
    except:
        return None

def normalize_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    return s if s else None

def localname(tag: str) -> str:
    # {namespace}tag -> tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def find_first(root: ET.Element, path_locals: list[str]) -> str | None:
    """
    Busca um caminho por "localnames" ignorando namespace, ex:
    ["infCte", "ide", "dhEmi"]
    """
    nodes = [root]
    for name in path_locals:
        next_nodes = []
        for n in nodes:
            for ch in list(n):
                if localname(ch.tag) == name:
                    next_nodes.append(ch)
        nodes = next_nodes
        if not nodes:
            return None
    # retorna texto do primeiro nó
    txt = nodes[0].text if nodes else None
    return normalize_text(txt)

def find_text_anywhere(root: ET.Element, tag_local: str) -> str | None:
    for el in root.iter():
        if localname(el.tag) == tag_local:
            if el.text and el.text.strip():
                return el.text.strip()
    return None


# -----------------------------
# Parsing CT-e XML
# -----------------------------
def extract_cte_key(xml_bytes: bytes) -> tuple[str, str]:
    """
    Retorna (cte_key, method):
      - method = "chCTe" quando consegue extrair 44 dígitos
      - method = "hash" quando usa hash do conteúdo
    """
    try:
        root = ET.fromstring(xml_bytes)
        ch = find_text_anywhere(root, "chCTe")
        if ch and re.fullmatch(r"\d{44}", ch):
            return ch, "chCTe"
    except ET.ParseError:
        pass

    return sha256_hex(xml_bytes), "hash"

def extract_fields(xml_bytes: bytes) -> dict:
    """
    Extrai campos relevantes com fallback pragmático (sem travar o pipeline).
    """
    out = {
        "emit_cnpj": None,
        "vTPrest": None,
        "dhEmi": None,
        "toma_uf": None,
    }
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out

    # emit/CNPJ
    # Estrutura típica: infCte/emit/CNPJ
    cnpj = find_first(root, ["infCte", "emit", "CNPJ"])
    if not cnpj:
        # fallback: busca CNPJ no documento (pode aparecer em outros nós)
        cnpj = find_text_anywhere(root, "CNPJ")
    out["emit_cnpj"] = normalize_text(cnpj)

    # vPrest/vTPrest
    # infCte/vPrest/vTPrest
    v = find_first(root, ["infCte", "vPrest", "vTPrest"])
    out["vTPrest"] = to_float_br(v)

    # dhEmi
    # infCte/ide/dhEmi (às vezes dEmi + hEmi em alguns layouts, mas hoje dhEmi é comum)
    dh = find_first(root, ["infCte", "ide", "dhEmi"])
    if not dh:
        # fallback: dEmi (data)
        dh = find_first(root, ["infCte", "ide", "dEmi"])
    out["dhEmi"] = normalize_text(dh)

    # toma/UF (toma4 é o mais comum; mas varia)
    # Possíveis caminhos:
    # infCte/ide/toma4/UF
    # infCte/ide/toma3/toma/UF
    # infCte/compl/enderToma/UF
    uf = find_first(root, ["infCte", "ide", "toma4", "UF"])
    if not uf:
        uf = find_first(root, ["infCte", "ide", "toma3", "toma", "UF"])
    if not uf:
        uf = find_first(root, ["infCte", "compl", "enderToma", "UF"])
    if not uf:
        # fallback: destino
        uf = find_first(root, ["infCte", "dest", "enderDest", "UF"])
    out["toma_uf"] = normalize_text(uf)

    return out


# -----------------------------
# DB (com migração leve)
# -----------------------------
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


# -----------------------------
# Upload handling
# -----------------------------
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


# -----------------------------
# Core processing
# -----------------------------
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
            emit_cnpj = fields["emit_cnpj"]
            vTPrest = fields["vTPrest"]
            dhEmi = fields["dhEmi"]
            toma_uf = fields["toma_uf"]

            row = con.execute("SELECT * FROM cte WHERE cte_key = ?", (cte_key,)).fetchone()

            if row is None:
                # novo
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
                        emit_cnpj, vTPrest, dhEmi, toma_uf, instituicao_financeira
                    ),
                )
                con.execute(
                    """
                    INSERT INTO batch_items(batch_id, cte_key, filename, is_duplicate, reason)
                    VALUES (?, ?, ?, 0, ?)
                    """,
                    (batch_id, cte_key, filename, f"novo ({method})"),
                )
                results.append(
                    {
                        "filename": filename,
                        "cte_key": cte_key,
                        "status_lote": "NOVO",
                        "status_atual": default_status,
                        "metodo_id": method,
                        "emit_cnpj": emit_cnpj,
                        "vTPrest": vTPrest,
                        "dhEmi": dhEmi,
                        "toma_uf": toma_uf,
                        "instituicao_financeira": instituicao_financeira,
                        "first_seen_batch_id": batch_id,
                        "first_seen_at_utc": now,
                    }
                )
            else:
                # duplicado
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

                results.append(
                    {
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
                    }
                )

        con.commit()

    return pd.DataFrame(results)


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


def update_status(cte_keys: list[str], new_status: str):
    if not cte_keys:
        return 0
    with sqlite3.connect(DB_PATH) as con:
        q_marks = ",".join(["?"] * len(cte_keys))
        params = [new_status] + cte_keys
        cur = con.execute(f"UPDATE cte SET status = ? WHERE cte_key IN ({q_marks})", params)
        con.commit()
        return cur.rowcount


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
ensure_db()

st.title(APP_TITLE)
st.caption("Upload de XML/ZIP → extração de campos → dedupe → histórico + workflow (PENDENTE/OPERADO/CANCELADO).")

with st.sidebar:
    st.subheader("Parâmetros operacionais")
    operator = st.text_input("Operador (opcional)", value="")
    notes = st.text_area("Observações (opcional)", value="", height=80)

    st.subheader("Instituição financeira")
    inst = st.text_input("Instituição (ex: Itaú, Bradesco, Santander)", value="")

    st.subheader("Status default para novos CT-e")
    default_status = st.selectbox("Status inicial", STATUS_VALUES, index=0)

    st.divider()
    st.subheader("Ações rápidas")
    key_query = st.text_input("Consultar CT-e por chave (44 dígitos ou hash)", value="")
    if st.button("Consultar", use_container_width=True):
        if key_query.strip():
            with sqlite3.connect(DB_PATH) as con:
                con.row_factory = sqlite3.Row
                r = con.execute("SELECT * FROM cte WHERE cte_key = ?", (key_query.strip(),)).fetchone()
            if r:
                st.success("Encontrado no histórico.")
                st.json(dict(r))
            else:
                st.warning("Não encontrado.")

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
        st.subheader("Resultado do lote (com campos extraídos)")
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
st.dataframe(
    df_cte[[
        "cte_key", "status", "emit_cnpj", "vTPrest", "dhEmi", "toma_uf",
        "instituicao_financeira", "first_seen_at_utc", "first_seen_batch_id"
    ]] if not df_cte.empty else df_cte,
    use_container_width=True,
    hide_index=True
)

st.markdown("**Atualização de status**: selecione chaves e aplique mudança com trilha explícita (sem apagar histórico).")

keys_to_change = []
if not df_cte.empty:
    sample_keys = df_cte["cte_key"].head(50).tolist()
    keys_to_change = st.multiselect(
        "Selecione CT-e (mostra os 50 primeiros do filtro acima)",
        options=sample_keys
    )

new_status = st.selectbox("Novo status", STATUS_VALUES, index=1)  # default OPERADO

if st.button("Aplicar novo status", disabled=not keys_to_change):
    changed = update_status(keys_to_change, new_status)
    st.success(f"Atualizados: {changed} CT-e para status {new_status}. Refiltre para atualizar a visão.")

st.caption(
    "Nota de governança: SQLite é ok para MVP no Streamlit Cloud. Para operação 'sem risco de reset', migre para Postgres externo."
)
