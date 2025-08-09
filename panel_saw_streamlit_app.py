import streamlit as st
from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv
import io
from collections import Counter

# ---------- Data classes ----------
@dataclass
class Piece:
    id: str
    w: float
    h: float
    qty: int = 1
    grain_locked: bool = False
    material: str = ""
    thickness_mm: Optional[float] = None

@dataclass
class Sheet:
    id: str
    w: float
    h: float
    qty: int = 1
    material: str = ""
    thickness_mm: Optional[float] = None

@dataclass
class Placement:
    piece_id: str
    sheet_id: str
    x: float
    y: float
    w: float
    h: float
    rotated: bool

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

# ---------- Guillotine packer ----------
class GuillotineBin:
    def __init__(self, w, h, kerf=0.0):
        self.w = w
        self.h = h
        self.kerf = kerf
        self.free: List[Rect] = [Rect(0, 0, w, h)]
        self.placements: List[Placement] = []

    def try_place_piece(self, pid, pw, ph, sheet_id, piece_allow_rotation=True) -> Optional[Placement]:
        kw = pw + self.kerf
        kh = ph + self.kerf

        for i, r in enumerate(self.free):
            # Try without rotation
            if kw <= r.w + 1e-9 and kh <= r.h + 1e-9:
                pl = Placement(pid, sheet_id, r.x, r.y, pw, ph, False)
                self._split_free(i, r, kw, kh)
                self.placements.append(pl)
                return pl
            # Try with rotation if allowed for this piece
            if piece_allow_rotation and (ph + self.kerf) <= r.w + 1e-9 and (pw + self.kerf) <= r.h + 1e-9:
                pl = Placement(pid, sheet_id, r.x, r.y, ph, pw, True)
                self._split_free(i, r, ph + self.kerf, pw + self.kerf)
                self.placements.append(pl)
                return pl
        return None

    def _split_free(self, index: int, rect: Rect, used_w: float, used_h: float):
        used_w = min(used_w, rect.w)
        used_h = min(used_h, rect.h)
        x0, y0 = rect.x, rect.y
        if rect.w - used_w > 1e-9:
            right = Rect(x0 + used_w, y0, rect.w - used_w, used_h)
            self.free.append(right)
        if rect.h - used_h > 1e-9:
            bottom = Rect(x0, y0 + used_h, rect.w, rect.h - used_h)
            self.free.append(bottom)
        del self.free[index]
        self._merge_free()

    def _merge_free(self):
        new_free = []
        for r in self.free:
            contained = False
            for s in self.free:
                if s is r:
                    continue
                if r.x >= s.x - 1e-9 and r.y >= s.y - 1e-9 and r.x + r.w <= s.x + s.w + 1e-9 and r.y + r.h <= s.y + s.h + 1e-9:
                    contained = True
                    break
            if not contained:
                new_free.append(r)
        self.free = new_free

# ---------- Optimizer ----------
def optimize(pieces: List[Piece], sheets: List[Sheet], kerf=3.0, allow_rotation_global=True) -> Tuple[List[Placement], List[Tuple[str,int]], List[Tuple[str,float,float]]]:
    singles = []
    for p in pieces:
        for i in range(max(1, p.qty)):
            singles.append(Piece(p.id + f"_{i+1}", p.w, p.h, qty=1, grain_locked=p.grain_locked))
    singles.sort(key=lambda z: (max(z.w, z.h), z.w * z.h), reverse=True)

    placements: List[Placement] = []
    bins: List[Tuple[GuillotineBin, str, str, Optional[float]]] = []
    for s in sheets:
        for i in range(max(1, s.qty)):
            sid = s.id + f"_{i+1}"
            bins.append((GuillotineBin(s.w, s.h, kerf=kerf), sid, getattr(s, 'material', ''), getattr(s, 'thickness_mm', None)))

    unplaced = []
    for piece in singles:
        placed = False
        piece_allow_rotation = (allow_rotation_global and not piece.grain_locked)
        for bin_obj, sid, smat, sth in bins:
            # if piece has a material/thickness, enforce match
            if (piece.material and piece.material.strip()) and (piece.material.strip() != (smat or '').strip()):
                continue
            if (piece.thickness_mm is not None) and (piece.thickness_mm != sth):
                continue
            pl = bin_obj.try_place_piece(piece.id, piece.w, piece.h, sid, piece_allow_rotation=piece_allow_rotation)
            if pl:
                placements.append(pl)
                placed = True
                break
        if not placed:
            unplaced.append(piece)

    unplaced_summary = {}
    for p in unplaced:
        base = "_".join(p.id.split("_")[:-1]) if "_" in p.id else p.id
        unplaced_summary[base] = unplaced_summary.get(base, 0) + 1

    # Collect remnant rectangles (free spaces) from each bin
    remnants: List[Tuple[str, float, float]] = []
    for bin_obj, sid in bins:
        for r in getattr(bin_obj, 'free', []):
            if r.w > 0.5 and r.h > 0.5:  # avoid tiny slivers
                remnants.append((sid, float(r.w), float(r.h)))
    return placements, list(unplaced_summary.items()), remnants

# ---------- CSV / SVG helpers ----------


def export_opal2070(parts: List[Piece],
                    sheets: List[Sheet],
                    panel_len: int,
                    panel_wid: int,
                    trim_L1: int = 0,
                    trim_L2: int = 0,
                    trim_W1: int = 0,
                    trim_W2: int = 0,
                    priority: int = 100,
                    date_str: str = "",
                    machine_name: str = "Opal2070") -> str:
    """
    Build an Opal2070 import file from parts.
    - Line 1 and 2 are fixed defaults.
    - Line 3: F -1 <panel_len> <panel_wid> 9999 <priority> <trim_L1> <trim_L2> <trim_W1> <trim_W2>
    - Lines 4-6: fixed defaults.
    - E-lines: aggregated by (L,B,grain_flag). Quantity is total occurrences.
      Grain triple: "1 1 1" if no grain; "1 1 0" if grain-locked.
    Units are millimeters.
    """
    from datetime import datetime
    if not date_str:
        date_str = datetime.now().strftime("%d.%m.%Y")

    # Header
    lines = []
    lines.append("\"Opal2070\" 1 \"***\" 190 \"34764\" 100 e")
    lines.append("A 0 e")
    lines.append(f"F -1 {int(panel_len)} {int(panel_wid)} 9999 {int(priority)} {int(trim_L1)} {int(trim_L2)} {int(trim_W1)} {int(trim_W2)} ")
    lines.append("F 0 0 0 0 0 0 0 0 0 ")
    lines.append("1 4 0 0 0 0 0 0 5000 125 0 0 0 0 440 32000 32000 1 0 0")

    # Aggregate parts: key = (L, B, grain_flag)
    from collections import defaultdict
    agg = defaultdict(int)
    for p in parts:
        qty = max(1, int(p.qty))
        L = int(round(max(p.w, 0)))
        B = int(round(max(p.h, 0)))
        # grain_flag: 1 if no grain, 0 if grain-locked (as per user's spec)
        grain_flag = 0 if getattr(p, "grain_locked", False) else 1
        agg[(L, B, grain_flag)] += qty

    # Emit one E-line per (L,B,grain_flag)
    seq = 19  # start like the sample
    idx = 1
    for (L, B, gf), q in agg.items():
        lines.append(
            f"E {seq} {idx} \"1\" {q} {L} {B} 0 0 0 0 0 0 0 0 2 2 2 2 {L} {B} 1 1 {gf} "
            + "\"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" "
            + f"\"{date_str}\" \"1010\" 0 0 0 0 6 6 6 6 3 3 3 3 "
            + "\"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" 0 0 0 0 0 0 0 0"
        )
        seq += 1
        idx += 1

    return "\n".join(lines) + "\n"

def generate_cutlist_csv(placements: List[Placement]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["piece_id","sheet_id","x_mm","y_mm","w_mm","h_mm","rotated"])
    for p in placements:
        writer.writerow([p.piece_id, p.sheet_id, f"{p.x:.2f}", f"{p.y:.2f}", f"{p.w:.2f}", f"{p.h:.2f}", p.rotated])
    return output.getvalue()

def generate_svg(placements: List[Placement], sheet_w=2440, sheet_h=1220) -> str:
    scale = 800.0 / sheet_w if sheet_w>0 else 1.0
    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet_w*scale}" height="{sheet_h*scale}" viewBox="0 0 {sheet_w*scale} {sheet_h*scale}">')
    svg_parts.append(f'<rect x="0" y="0" width="{sheet_w*scale}" height="{sheet_h*scale}" fill="#ffffff" stroke="#000" stroke-width="1"/>')
    for p in placements:
        x = p.x * scale
        y = p.y * scale
        w = p.w * scale
        h = p.h * scale
        color = "#%06x" % (abs(hash(p.piece_id)) & 0xFFFFFF)
        svg_parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{color}" fill-opacity="0.25" stroke="#000" stroke-width="0.5"/>')
        svg_parts.append(f'<text x="{x+2:.2f}" y="{y+12:.2f}" font-size="10">{p.piece_id}</text>')
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def export_panhans_ncr(parts: List[Piece],
                       sheets: List[Sheet],
                       panel_len: int,
                       panel_wid: int,
                       job_name: str = "Job",
                       program_name: str = "JOB.S01",
                       customer: str = "",
                       kerf_hundredths: int = 190,   # D00190 in HA line sample
                       date_dmy: str = "",
                       time_str: str = "",
                       auto_ids: bool = True) -> str:
    """
    Build a minimal Panhans NCR file from parts (header + I-lines).
    - Uses user's template lines for V100, Sc54, ZM000, RZ..., MF1, HA...
    - Generates I-lines: I<index> Z<index> L<len> B<wid> H<len> W<wid> s<qty> ... w\"dd.mm.YYYY\" ... o<qty> e
    - Does NOT generate A/K cut path blocks (can be added later).
    Units are millimeters. Kerf in hundredths (D00190 = 1.90 mm) configurable.
    """
    from datetime import datetime
    if not date_dmy:
        date_dmy = datetime.now().strftime("%d.%m.%Y")
    if not time_str:
        time_str = datetime.now().strftime("%d/%m/%Y/%H:%M")  # t\"8/9/2025/09:08\" style (d/m/Y/H:M)

    lines = []
    lines.append('7')  # file marker as per sample
    lines.append('V100 v\"1.0056 13.08.2007\" S9 D4 R5000 125 B0 0 0 0 0 e')
    lines.append('Sc54 d17 b4 p4 s4 l87718 F23184 0 21680 23184 e')
    lines.append('ZM000 K00012000 T0250 t0150 w0200 S0200 s0700 B0250 b0700 O15000 R00400 k03600 m09500 Z0015 z0005 D0025 W0020 H0015 h0050 A0000 r0030 P0005 C0000 F0000 e')
    lines.append('RZ00001509 K00000005029 L0000087480 S00001069 H00000440 B00000100 z00000000 k00000000000 l0000000000 f0000000000 s00000000 h00000000 b00000000 p00000000 g00000000000 d0000000000 c00000000 w00000000 a00000000 e')
    lines.append(f'MF1 L{int(panel_len)} B{int(panel_wid)} s9999 i4 a3 v0 h0 l0 r0 p100 T0 e')
    lines.append(f'HA\"34764.NCR\" a\"{customer or job_name}\" m\"***\" D{kerf_hundredths:05d} s44 M1 S001 f\"100100101010110000010000100000000\" b\"{customer or job_name}\" p\"***\" t\"{time_str}\" v0 h0 l0 r0 I\"                    \" N4400 V0200 n\"01\" U7000 E\"\" P\"{program_name}\" Z0000 g0 e')

    # Aggregate identical parts (L,B) by qty
    from collections import defaultdict
    agg = defaultdict(int)
    for p in parts:
        qty = max(1, int(p.qty))
        L = int(round(max(p.w, 0)))
        B = int(round(max(p.h, 0)))
        agg[(L, B)] += qty

    # I-lines
    idx = 1
    for (L, B), q in agg.items():
        lines.append(
            f'I{idx} Z{idx} L{L} B{B} H{L} W{B} s{q} u\"\" p\"1\" z\"\" v\"\" h\"\" l\"\" r\"\" i\"\" a\"\" w\"{date_dmy}\" '
            + 'd0 0 0 0 k2 2 2 2 K6 6 6 6 3 3 3 3 S00000 U0 0 0 0 F\"01000\" E\"{0}\" x\"\" '  # E\"{0}\" left literal like sample
            + '\"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" \"\" g0 0 0 0 0 0 0 0 t0 0 0 0 '
            + f'o{q} e'
        )
        idx += 1

    # For now skip T/A/K cut-path blocks; end program
    lines.append('E!')
    return "\n".join(lines) + "\n"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Panel Saw Cutting Optimizer", layout="wide")
st.title("Panel Saw Cutting Optimizer — Guillotine 2D with Grain Lock")

with st.sidebar:
    st.header("Inputs")
    uploaded_parts = st.file_uploader("Upload parts CSV (id,width_mm,height_mm,qty,grain_locked)", type=["csv"])
    uploaded_sheets = st.file_uploader("Upload sheets CSV (id,width_mm,height_mm,qty)", type=["csv"])
    kerf = st.number_input("Kerf (mm)", value=3.0, min_value=0.0, step=0.1)
    allow_rotation_global = st.checkbox("Allow rotation globally", value=True)
    run_btn = st.button("Run optimizer")

with st.sidebar.expander("Opal2070 export options"):
    st.caption("Header values and trim cuts (mm)")
    auto_swap = st.checkbox("Auto swap panel L/W (mirror orientation)", value=True)
    force_length_along_feed = st.checkbox("Force part length along feed direction", value=True)
    default_len = int(sheets[0].w) if sheets else 28000
    default_wid = int(sheets[0].h) if sheets else 20700
    opal_len = st.number_input("Panel length (mm)", value=default_len, step=10)
    opal_wid = st.number_input("Panel width (mm)", value=default_wid, step=10)
    opal_priority = st.number_input("Priority", value=100, step=1)
    c1, c2 = st.columns(2)
    with c1:
        opal_trim_L1 = st.number_input("Trim length side 1 (mm)", value=0, step=1)
        opal_trim_W1 = st.number_input("Trim width side 1 (mm)", value=0, step=1)
    with c2:
        opal_trim_L2 = st.number_input("Trim length side 2 (mm)", value=0, step=1)
        opal_trim_W2 = st.number_input("Trim width side 2 (mm)", value=0, step=1)


st.markdown("""
**CSV format examples**

Parts CSV columns: `id,width_mm,height_mm,qty[,grain_locked,material,thickness_mm]` (optional fields in brackets)  
Sheets CSV columns: `id,width_mm,height_mm,qty[,material,thickness_mm]`  
""")

# Example CSV text
example_parts_text = "id,width_mm,height_mm,qty,grain_locked,material,thickness_mm\nA,800,600,2,True,MDF,18\nB,400,300,6,False,Particleboard,18\nC,1200,500,1,True,MDF,18\nD,600,400,3,False,Plywood,12\n"
example_sheets_text = "id,width_mm,height_mm,qty,material,thickness_mm\nSHEET_1,2440,1220,3,Particleboard,18\nSHEET_2,2800,2070,1,MDF,18\n"

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download example parts CSV", data=example_parts_text, file_name="example_parts.csv")
with col2:
    st.download_button("Download example sheets CSV", data=example_sheets_text, file_name="example_sheets.csv")

# Parsing functions
def parse_parts_csv(file) -> List[Piece]:
    if hasattr(file, "read"):
        text = file.read().decode("utf-8")
    else:
        text = file
    reader = csv.DictReader(io.StringIO(text))
    parts = []
    for r in reader:
        grain_val = str(r.get("grain_locked","False")).strip().lower() in ("true","1","yes","y")
        mat = (r.get("material", "") or "").strip()
        th = r.get("thickness_mm", None)
        th_val = float(th) if th not in (None, "") else None
        parts.append(Piece(
            r["id"],
            float(r["width_mm"]),
            float(r["height_mm"]),
            int(r.get("qty",1)),
            grain_locked=grain_val,
            material=mat,
            thickness_mm=th_val
        ))
    return parts

def parse_sheets_csv(file) -> List[Sheet]:
    if hasattr(file, "read"):
        text = file.read().decode("utf-8")
    else:
        text = file
    reader = csv.DictReader(io.StringIO(text))
    sheets = []
    for r in reader:
        material = (r.get("material", "") or "").strip()
        th = r.get("thickness_mm", None)
        th_val = float(th) if th not in (None, "") else None
        sheets.append(Sheet(
            r["id"],
            float(r["width_mm"]),
            float(r["height_mm"]),
            int(r.get("qty",1)),
            material=material,
            thickness_mm=th_val
        ))
    return sheets



def parse_remnants_csv(file):
    """Return a list of dicts with keys:
       id (optional), width_mm, height_mm, qty, material (optional), thickness_mm (optional)
    """
    import csv, io
    if hasattr(file, 'read'):
        text = file.read().decode('utf-8')
    else:
        text = file
    reader = csv.DictReader(io.StringIO(text))
    rems = []
    for r in reader:
        rec = {
            "id": r.get("id", "").strip(),
            "width_mm": float(r["width_mm"]),
            "height_mm": float(r["height_mm"]),
            "qty": int(r.get("qty", 1)),
            "material": (r.get("material", "") or "").strip(),
            "thickness_mm": float(r["thickness_mm"]) if r.get("thickness_mm") not in (None, "",) else None,
        }
        rems.append(rec)
    return rems

# Load defaults if no upload
if not uploaded_parts:
    st.info("No parts CSV uploaded — using example data.")
    parts = parse_parts_csv(example_parts_text)
else:
    uploaded_parts.seek(0)
    parts = parse_parts_csv(uploaded_parts)

if not uploaded_sheets:
    sheets = parse_sheets_csv(example_sheets_text)
else:
    uploaded_sheets.seek(0)
    sheets = parse_sheets_csv(uploaded_sheets)

# Show preview
st.subheader("Parts (preview)")
parts_table = [[p.id, p.w, p.h, p.qty, ('🔒' if p.grain_locked else '↻'), p.material, p.thickness_mm] for p in parts]
st.table(parts_table)

st.subheader("Sheets (preview)")
sheets_table = [[s.id, s.w, s.h, s.qty, s.material, s.thickness_mm] for s in sheets]
st.table(sheets_table)

if run_btn:
    with st.spinner("Running optimizer..."):
        placements, unplaced, remnants = optimize(parts, sheets, kerf=kerf, allow_rotation_global=allow_rotation_global)
    st.success(f"Done — placed {len(placements)} pieces. Unplaced types: {len(unplaced)}")
    # Safety check: grain-locked pieces should not be rotated
    grain_map = {pp.id: pp.grain_locked for pp in parts}
    def _base_id2(pid: str):
        return "_".join(pid.split("_")[:-1]) if "_" in pid else pid
    violations = [p for p in placements if p.rotated and grain_map.get(_base_id2(p.piece_id), False)]
    if violations:
        ids = sorted({p.piece_id for p in violations})
        st.error(f"Warning: {len(violations)} placement(s) show rotated parts that are grain-locked: {', '.join(ids)}")

    # Material/thickness mismatch check (defensive)
    try:
        sheet_meta = {s.id: (getattr(s,'material',''), getattr(s,'thickness_mm', None)) for s in sheets}
        mism = []
        def _base_id3(pid: str):
            return "_".join(pid.split("_")[:-1]) if "_" in pid else pid
        part_meta = {pp.id: (pp.material, pp.thickness_mm) for pp in parts}
        for p in placements:
            mat_th = part_meta.get(_base_id3(p.piece_id), ("", None))
            smt = sheet_meta.get(p.sheet_id, ("", None))
            if (mat_th[0] and mat_th[0].strip()) and (mat_th[0].strip() != (smt[0] or '').strip()):
                mism.append(p.piece_id)
            if (mat_th[1] is not None) and (mat_th[1] != smt[1]):
                mism.append(p.piece_id)
        if mism:
            st.error(f"Material/thickness mismatch on placements: {', '.join(sorted(set(mism)))}")
    except Exception:
        pass

    cnt = Counter([p.sheet_id for p in placements])
    st.write("Sheets used (counts):", dict(cnt))
    # Grouping by material/thickness
    try:
        by_mat = {}
        sheet_meta = {}
        for s in sheets:
            sheet_meta[s.id] = (getattr(s, 'material',''), getattr(s, 'thickness_mm', None))
        for p in placements:
            mat, th = sheet_meta.get(p.sheet_id.split('_')[0], sheet_meta.get(p.sheet_id, ('','')))
            key = f"{mat} / {'' if th is None else str(int(th))+'mm'}"
            by_mat[key] = by_mat.get(key, 0) + 1
        st.write("By material/thickness (pieces):", by_mat)
    except Exception:
        pass
    if unplaced:
        st.warning("Unplaced items (counts): " + str(dict(unplaced)))

        # Remnants export
    rem_rows = []
    for sid, w, h in remnants:
        if w >= min_remnant_w and h >= min_remnant_h:
            rem_rows.append({"sheet_id": sid, "width_mm": round(w,1), "height_mm": round(h,1)})
    remnants_df = pd.DataFrame(rem_rows)
    if not remnants_df.empty:
        st.subheader("Remnants (filtered)")
        st.dataframe(remnants_df)
        rem_csv = remnants_df.to_csv(index=False)
        st.download_button("Download remnants CSV", rem_csv, file_name="remnants.csv", mime="text/csv")
    else:
        st.info("No remnants meeting the size filter.")

st.subheader("Placements (first 200 rows)")
    rows = []
    # Build lookup for grain lock from parts
    grain_lookup = {p.id: p.grain_locked for p in parts}
    for pl in placements[:200]:
        base_id = pl.piece_id.split("_")[0]
        grain_val = grain_lookup.get(base_id, False)
        grain_icon = "✅" if grain_val else ""
        rows.append([pl.piece_id, grain_icon, pl.sheet_id, f"{pl.x:.1f}", f"{pl.y:.1f}", f"{pl.w:.1f}", f"{pl.h:.1f}", pl.rotated])
    st.table([["Piece ID", "Grain Locked", "Sheet ID", "X (mm)", "Y (mm)", "W (mm)", "H (mm)", "Rotated"]] + rows)

    csv_text = generate_cutlist_csv(placements)
    svg_text = generate_svg(placements, sheet_w=(sheets[0].w if sheets else 2440), sheet_h=(sheets[0].h if sheets else 1220))

    st.download_button("Download cutlist CSV", csv_text, file_name="cutlist.csv", mime="text/csv")
    st.download_button("Download cutplan SVG", svg_text, file_name="cutplan.svg", mime="image/svg+xml")

    
    # --- Build DataFrames for Excel export ---
    parts_df = pd.DataFrame([{"id": p.id, "width_mm": p.w, "height_mm": p.h, "qty": p.qty, "grain_locked": p.grain_locked} for p in parts])
    sheets_df = pd.DataFrame([{"id": s.id, "width_mm": s.w, "height_mm": s.h, "qty": s.qty, "material": s.material, "thickness_mm": s.thickness_mm} for s in sheets])
    placements_df = pd.DataFrame([
        {"piece_id": p.piece_id, "sheet_id": p.sheet_id, "x_mm": round(p.x,1), "y_mm": round(p.y,1), "w_mm": round(p.w,1), "h_mm": round(p.h,1), "rotated": p.rotated}
        for p in placements
    ])

    # Simple cut sequence: order by sheet_id, then y, then x (top-left -> bottom-right)
    if not placements_df.empty:
        placements_df = placements_df.sort_values(by=["sheet_id","y_mm","x_mm"]).reset_index(drop=True)
        placements_df.insert(0, "seq", placements_df.index + 1)

    summary_rows = [
        {"metric": "Placed pieces", "value": len(placements)},
        {"metric": "Unplaced types", "value": len(unplaced)},
        {"metric": "Kerf (mm)", "value": kerf},
        {"metric": "Global rotation allowed", "value": allow_rotation_global},
    ]
    if unplaced:
        for k,v in dict(unplaced).items():
            summary_rows.append({"metric": f"Unplaced {k}", "value": v})
    summary_df = pd.DataFrame(summary_rows)

    # Try to build an Excel workbook
    excel_bytes = None
    try:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            parts_df.to_excel(writer, index=False, sheet_name="Parts")
            sheets_df.to_excel(writer, index=False, sheet_name="Sheets")
            placements_df.to_excel(writer, index=False, sheet_name="Placements")
        bio.seek(0)
        excel_bytes = bio.read()
        st.download_button("Download Excel workbook (.xlsx)", excel_bytes, file_name="cutting_plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e1:
        try:
            import xlsxwriter  # noqa: F401
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
                summary_df.to_excel(writer, index=False, sheet_name="Summary")
                parts_df.to_excel(writer, index=False, sheet_name="Parts")
                sheets_df.to_excel(writer, index=False, sheet_name="Sheets")
                placements_df.to_excel(writer, index=False, sheet_name="Placements")
            bio.seek(0)
            excel_bytes = bio.read()
            st.download_button("Download Excel workbook (.xlsx)", excel_bytes, file_name="cutting_plan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e2:
            # Fallback: ZIP of CSVs
            zip_bio = io.BytesIO()
            with zipfile.ZipFile(zip_bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("Summary.csv", summary_df.to_csv(index=False))
                zf.writestr("Parts.csv", parts_df.to_csv(index=False))
                zf.writestr("Sheets.csv", sheets_df.to_csv(index=False))
                zf.writestr("Placements.csv", placements_df.to_csv(index=False))
            zip_bio.seek(0)
            st.warning("Excel engine not available; offering a ZIP of CSVs instead.")
            st.download_button("Download results (ZIP of CSVs)", zip_bio.read(), file_name="cutting_plan_csvs.zip", mime="application/zip")


    st.subheader("Cut plan preview (SVG)")
    st.components.v1.html(svg_text, height=600, scrolling=True)
    st.markdown(
    '<div style="margin-top:10px;">'
    '<span style="display:inline-block;width:20px;height:20px;background-color:lightcoral;margin-right:5px;border:1px solid #000;"></span> Grain-locked (no rotation)'
    '&nbsp;&nbsp;&nbsp;'
    '<span style="display:inline-block;width:20px;height:20px;background-color:lightblue;margin-right:5px;border:1px solid #000;"></span> Rotation allowed'
    '</div>', unsafe_allow_html=True
)

st.markdown("---")
st.markdown("Now supports per-piece grain lock: set `grain_locked` to True in the parts CSV to forbid rotation for that piece.")
