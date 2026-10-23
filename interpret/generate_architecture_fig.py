import os
os.environ["MPLCONFIGDIR"] = "/Users/zengrao/.gemini/antigravity/brain/4bd4252e-5830-4360-b443-c13835d29da9/scratch/mpl"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# Configure publication-grade Matplotlib settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'stixsans'

BASE_DIR = "/Volumes/Files/Research/PhyNet-DTA"
FIG_DIR = os.path.join(BASE_DIR, "paper", "figures")
OUT_PDF = os.path.join(BASE_DIR, "paper", "architecture.pdf")
OUT_PNG = os.path.join(FIG_DIR, "architecture.png")
ARTIFACT_PNG = "/Users/zengrao/.gemini/antigravity/brain/4bd4252e-5830-4360-b443-c13835d29da9/architecture_preview.png"

FIG_W = 13.8
FIG_H = 8.5
FIG_ASPECT = FIG_W / FIG_H


def add_rounded_box(ax, x, y, w, h, facecolor="#FFFFFF", edgecolor="#CBD5E1",
                    lw=1.2, radius=0.012, alpha=1.0, zorder=2, linestyle="-",
                    shadow=True):
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.0020, y - 0.0028), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            facecolor="#334155", edgecolor="none", alpha=0.035,
            zorder=zorder - 0.2, transform=ax.transData
        )
        ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        alpha=alpha, linestyle=linestyle, zorder=zorder,
        transform=ax.transData
    )
    ax.add_patch(box)
    return box


def add_header_banner(ax, x, y, w, h, title, facecolor="#E2E8F0",
                      textcolor="#1E293B", edgecolor="#CBD5E1", radius=0.010, fontsize=11.6, zorder=4):
    banner = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.1, zorder=zorder,
        transform=ax.transData
    )
    ax.add_patch(banner)
    ax.text(x + w / 2, y + h * 0.50, title, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=textcolor, zorder=zorder + 1)


def add_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.9, style="-|>",
              mutation_scale=13, connectionstyle="arc3,rad=0.0", linestyle="-", zorder=6):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        zorder=zorder,
        transform=ax.transData
    )
    ax.add_patch(arr)
    return arr


def draw_token_matrix(ax, x, y, w, h, n_rows, n_cols, row_labels, palette,
                      mask_rows=None, title="", dim_label="", zorder=5):
    """Draws a crisp token matrix grid with optional row labels and masked padding rows."""
    if mask_rows is None:
        mask_rows = []
    has_labels = any(bool(lbl) for lbl in row_labels)
    label_w = w * 0.33 if has_labels else 0.0
    grid_x = x + label_w
    grid_w = w - label_w
    cell_h = h / n_rows
    cell_w = grid_w / n_cols

    if title:
        title_cx = x + w / 2 if has_labels else grid_x + grid_w / 2
        ax.text(title_cx, y + h + 0.010, title, ha="center", va="bottom",
                fontsize=9.6, fontweight="bold", color="#1E293B", zorder=zorder + 1)

    for r in range(n_rows):
        ry = y + h - (r + 1) * cell_h
        is_masked = r in mask_rows
        if has_labels and r < len(row_labels):
            lbl = row_labels[r]
            lbl_color = "#64748B" if is_masked else "#1E293B"
            ax.text(x + label_w - 0.004, ry + cell_h / 2, lbl, ha="right", va="center",
                    fontsize=8.6, fontweight="bold",
                    color=lbl_color, zorder=zorder + 1)
        for c in range(n_cols):
            cx = grid_x + c * cell_w
            if is_masked:
                fc = "#EDF2F7"
                ec = "#E2E8F0"
                alpha = 0.65
            else:
                fc = palette[(r + c) % len(palette)]
                ec = "#FFFFFF"
                alpha = 0.78
            rect = patches.Rectangle(
                (cx + 0.0008, ry + 0.0012), cell_w - 0.0016, cell_h - 0.0024,
                facecolor=fc, edgecolor=ec, linewidth=0.6, alpha=alpha, zorder=zorder
            )
            ax.add_patch(rect)
            if is_masked and c == n_cols // 2:
                ax.text(cx + cell_w / 2, ry + cell_h / 2, "0", ha="center", va="center",
                        fontsize=7.6, fontweight="bold", color="#94A3B8", zorder=zorder + 1)

    if dim_label:
        ax.text(x + w / 2, y - 0.010, dim_label, ha="center", va="top",
                fontsize=9.2, color="#475569", fontweight="bold", zorder=zorder + 1)


def draw_vector_bar(ax, x, y, w, h, n_cells, colors, vertical=True, label="", sublabel="", zorder=5):
    """Draws a 1D feature vector as a segmented bar of colored channels."""
    add_rounded_box(ax, x, y, w, h, facecolor="#F8FAFC", edgecolor="#94A3B8",
                    lw=1.0, radius=0.005, shadow=False, zorder=zorder)
    if vertical:
        ch = (h - 0.008) / n_cells
        for i in range(n_cells):
            cy = y + h - 0.004 - (i + 1) * ch
            c = colors[i % len(colors)]
            rect = patches.Rectangle(
                (x + 0.003, cy + 0.0015), w - 0.006, ch - 0.003,
                facecolor=c, edgecolor="#FFFFFF", linewidth=0.5, alpha=0.80, zorder=zorder + 1
            )
            ax.add_patch(rect)
    else:
        cw = (w - 0.008) / n_cells
        for i in range(n_cells):
            cx = x + 0.004 + i * cw
            c = colors[i % len(colors)]
            rect = patches.Rectangle(
                (cx + 0.001, y + 0.003), cw - 0.002, h - 0.006,
                facecolor=c, edgecolor="#FFFFFF", linewidth=0.5, alpha=0.80, zorder=zorder + 1
            )
            ax.add_patch(rect)
    if label:
        ax.text(x + w / 2, y + h + 0.008, label, ha="center", va="bottom",
                fontsize=10.2, fontweight="bold", color="#1E293B", zorder=zorder + 2)
    if sublabel:
        ax.text(x + w / 2, y - 0.009, sublabel, ha="center", va="top",
                fontsize=9.0, fontweight="bold", color="#475569", zorder=zorder + 2)


def draw_mlp_icon(ax, x, y, w, h, layer_sizes, node_colors, r_y=0.0062, edge_color="#94A3B8", zorder=5):
    """Draws a multi-layer neural network diagram with true geometric vector circles (aspect-corrected)."""
    n_layers = len(layer_sizes)
    r_x = r_y / FIG_ASPECT
    xs = np.linspace(x + r_x + 0.006, x + w - r_x - 0.006, n_layers)
    coords = []
    for li, n_nodes in enumerate(layer_sizes):
        if n_nodes == 1:
            ys = [y + h / 2.0]
        else:
            pad_y = r_y + 0.008
            ys = np.linspace(y + pad_y, y + h - pad_y, n_nodes)
        coords.append([(xs[li], yv) for yv in ys])

    for li in range(n_layers - 1):
        for (x1, y1) in coords[li]:
            for (x2, y2) in coords[li + 1]:
                ax.plot([x1, x2], [y1, y2], color=edge_color, lw=0.95, alpha=0.55, zorder=zorder)

    for li, layer in enumerate(coords):
        col = node_colors[li % len(node_colors)]
        for (nx, ny) in layer:
            ell = patches.Ellipse(
                (nx, ny), width=2.0 * r_x, height=2.0 * r_y,
                facecolor=col, edgecolor="#475569", linewidth=1.0, zorder=zorder + 1
            )
            ax.add_patch(ell)
    return coords[-1][0]


def crop_and_place_image(ax, img_path, extent, crop_box=None, whiteout_boxes=None, zorder=4, border_color=None, bg_color="#FFFFFF"):
    """Loads an image, crops (l, t, r, b) in [0,1] fractions, and places it inside extent=[x0, x1, y0, y1] with 100% undistorted physical aspect ratio and native resolution."""
    if not os.path.exists(img_path):
        return
    im = Image.open(img_path).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    H, W, _ = arr.shape
    if whiteout_boxes:
        for (wl, wt, wr, wb) in whiteout_boxes:
            arr[int(wt * H):int(wb * H), int(wl * W):int(wr * W), :] = 255
    if crop_box is not None:
        l, t, r, b = crop_box
        arr = arr[int(t * H):int(b * H), int(l * W):int(r * W), :]

    ch_px, cw_px, _ = arr.shape
    img_aspect = cw_px / max(ch_px, 1)

    x0, x1, y0, y1 = extent
    dw = x1 - x0
    dh = y1 - y0
    if bg_color:
        bg = patches.Rectangle((x0, y0), dw, dh, facecolor=bg_color, edgecolor="none", zorder=zorder - 0.5)
        ax.add_patch(bg)

    box_aspect = (dw * FIG_ASPECT) / max(dh, 1e-6)
    if img_aspect >= box_aspect:
        new_dw = dw * 0.97
        new_dh = (new_dw * FIG_ASPECT) / img_aspect
    else:
        new_dh = dh * 0.97
        new_dw = (new_dh * img_aspect) / FIG_ASPECT

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    img_extent = [cx - new_dw / 2.0, cx + new_dw / 2.0, cy - new_dh / 2.0, cy + new_dh / 2.0]
    ax.imshow(arr, extent=img_extent, aspect="auto", interpolation="none", zorder=zorder)

    if border_color:
        rect = patches.Rectangle((x0, y0), dw, dh, facecolor="none",
                                 edgecolor=border_color, linewidth=1.0, zorder=zorder + 1)
        ax.add_patch(rect)


def build_architecture_figure():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
    ax = fig.add_axes([0.004, 0.004, 0.992, 0.992])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    pal_drug = ["#F5B041", "#F8C471", "#FAD7A0", "#EB984E", "#F1948A", "#F5B7B1"]
    pal_prot = ["#7FB3D5", "#85C1E9", "#5499C7", "#76D7C4", "#A3E4D7", "#A5B4FC"]
    pal_attn_l = ["#EB984E", "#F5B041", "#F8C471", "#76D7C4", "#F1948A"]
    pal_attn_p = ["#5499C7", "#7FB3D5", "#76D7C4", "#A5B4FC", "#85C1E9"]

    # =========================================================================
    # TOP TIER: STAGES 1, 2, AND 3 (Y = 0.460 to 0.988)
    # =========================================================================
    top_y0, top_h = 0.460, 0.528
    banner_h = 0.044
    banner_y = top_y0 + top_h - banner_h

    # --- STAGE 1: FROZEN FOUNDATION ENCODING (x = 0.008 to 0.338) ---
    s1_x, s1_w = 0.008, 0.330
    add_rounded_box(ax, s1_x, top_y0, s1_w, top_h,
                    facecolor="#FAFBFD", edgecolor="#D8E2EC", lw=1.3, radius=0.014, zorder=1)
    add_header_banner(ax, s1_x, banner_y, s1_w, banner_h,
                      "Stage 1  |  Frozen Foundation Encoding",
                      facecolor="#E6EDF5", textcolor="#2C3E50", edgecolor="#CBD5E1", fontsize=11.5)

    # 1A. Drug Stream Card (Top half of Stage 1)
    d_card_y, d_card_h = 0.708, 0.222
    add_rounded_box(ax, s1_x + 0.010, d_card_y, s1_w - 0.020, d_card_h,
                    facecolor="#FFFDF9", edgecolor="#F5CBA7", lw=1.15, radius=0.010, zorder=2)
    ax.text(s1_x + 0.018, d_card_y + d_card_h - 0.020,
            "Ligand Modality",
            ha="left", va="center", fontsize=10.2, fontweight="bold", color="#935116", zorder=4)

    # SMILES + 2D molecule mini-box
    add_rounded_box(ax, s1_x + 0.016, d_card_y + 0.118, 0.148, 0.068,
                    facecolor="#FFFFFF", edgecolor="#F5CBA7", lw=1.0, radius=0.006, shadow=False, zorder=3)
    afa_2d_path = os.path.join(FIG_DIR, "fig1_panel_b", "4G5J_EGFR_Afatinib", "4G5J_EGFR_Afatinib_3_2d_pharmacophore.png")
    crop_and_place_image(ax, afa_2d_path,
                         extent=[s1_x + 0.019, s1_x + 0.065, d_card_y + 0.121, d_card_y + 0.183],
                         crop_box=(0.12, 0.18, 0.88, 0.88), zorder=4)
    ax.text(s1_x + 0.114, d_card_y + 0.162, "SMILES String",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="#AF601A", zorder=4)
    ax.text(s1_x + 0.114, d_card_y + 0.138, "& 3D Conformer",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#935116", zorder=4)

    # Frozen Uni-Mol Encoder box
    add_rounded_box(ax, s1_x + 0.016, d_card_y + 0.012, 0.148, 0.088,
                    facecolor="#FEF5E7", edgecolor="#E59866", lw=1.15, radius=0.008, shadow=False, zorder=3)
    ax.text(s1_x + 0.090, d_card_y + 0.081, "Uni-Mol (Frozen)",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#935116", zorder=4)
    ax.text(s1_x + 0.090, d_card_y + 0.058, "3D SE(3) Encoder",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#784212", zorder=4)
    for ti in range(3):
        tx = s1_x + 0.028 + ti * 0.044
        add_rounded_box(ax, tx, d_card_y + 0.018, 0.036, 0.026,
                        facecolor="#FFFFFF", edgecolor="#E59866", lw=0.9, radius=0.004, shadow=False, zorder=4)
        ax.text(tx + 0.018, d_card_y + 0.031, f"L{ti+1}", ha="center", va="center",
                fontsize=8.4, fontweight="bold", color="#AF601A", zorder=5)

    add_arrow(ax, s1_x + 0.090, d_card_y + 0.117, s1_x + 0.090, d_card_y + 0.101,
              color="#D68910", lw=1.5, mutation_scale=11)
    add_arrow(ax, s1_x + 0.165, d_card_y + 0.056, s1_x + 0.184, d_card_y + 0.056,
              color="#D68910", lw=1.6, mutation_scale=12)

    # Atomic Token Matrix X_L
    add_rounded_box(ax, s1_x + 0.185, d_card_y + 0.012, 0.131, 0.182,
                    facecolor="#FFFFFF", edgecolor="#F5CBA7", lw=1.0, radius=0.007, shadow=False, zorder=3)
    draw_token_matrix(
        ax, s1_x + 0.189, d_card_y + 0.046, 0.123, 0.114,
        n_rows=6, n_cols=6,
        row_labels=["[CLS]", "C1", "N3", "Cl1", "C27", "[PAD]"],
        palette=pal_drug, mask_rows=[0, 5],
        title=r"Atom Tokens $X_L$",
        dim_label=r"$N_L \times 512$",
        zorder=4
    )

    # 1B. Protein Stream Card (Bottom half of Stage 1)
    p_card_y, p_card_h = 0.470, 0.222
    add_rounded_box(ax, s1_x + 0.010, p_card_y, s1_w - 0.020, p_card_h,
                    facecolor="#F7FBFE", edgecolor="#A9CCE3", lw=1.15, radius=0.010, zorder=2)
    ax.text(s1_x + 0.018, p_card_y + p_card_h - 0.020,
            "Target Modality",
            ha="left", va="center", fontsize=10.2, fontweight="bold", color="#1F618D", zorder=4)

    # FASTA + 3D ribbon mini-box
    add_rounded_box(ax, s1_x + 0.016, p_card_y + 0.120, 0.148, 0.068,
                    facecolor="#FFFFFF", edgecolor="#A9CCE3", lw=1.0, radius=0.006, shadow=False, zorder=3)
    pose_img = os.path.join(FIG_DIR, "fig1_panel_b", "4G5J_EGFR_Afatinib", "4G5J_EGFR_Afatinib_1_3d_pose.png")
    crop_and_place_image(ax, pose_img,
                         extent=[s1_x + 0.019, s1_x + 0.065, p_card_y + 0.123, p_card_y + 0.185],
                         crop_box=(0.05, 0.11, 0.95, 0.96), zorder=4)
    ax.text(s1_x + 0.114, p_card_y + 0.164, "FASTA Sequence",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="#2471A3", zorder=4)
    ax.text(s1_x + 0.114, p_card_y + 0.140, "(1D Primary Seq.)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#1F618D", zorder=4)

    # Frozen ESM-2 Encoder box
    add_rounded_box(ax, s1_x + 0.016, p_card_y + 0.012, 0.148, 0.088,
                    facecolor="#EBF5FB", edgecolor="#7FB3D5", lw=1.15, radius=0.008, shadow=False, zorder=3)
    ax.text(s1_x + 0.090, p_card_y + 0.081, "ESM-2 (Frozen)",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1F618D", zorder=4)
    ax.text(s1_x + 0.090, p_card_y + 0.058, "Protein LM (35M)",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#1B4F72", zorder=4)
    for ti in range(3):
        tx = s1_x + 0.028 + ti * 0.044
        add_rounded_box(ax, tx, p_card_y + 0.018, 0.036, 0.026,
                        facecolor="#FFFFFF", edgecolor="#7FB3D5", lw=0.9, radius=0.004, shadow=False, zorder=4)
        ax.text(tx + 0.018, p_card_y + 0.031, f"T{ti+1}", ha="center", va="center",
                fontsize=8.4, fontweight="bold", color="#2471A3", zorder=5)

    add_arrow(ax, s1_x + 0.090, p_card_y + 0.119, s1_x + 0.090, p_card_y + 0.101,
              color="#5499C7", lw=1.5, mutation_scale=11)
    add_arrow(ax, s1_x + 0.165, p_card_y + 0.056, s1_x + 0.184, p_card_y + 0.056,
              color="#5499C7", lw=1.6, mutation_scale=12)

    # Residue Token Matrix X_P
    add_rounded_box(ax, s1_x + 0.185, p_card_y + 0.012, 0.131, 0.182,
                    facecolor="#FFFFFF", edgecolor="#A9CCE3", lw=1.0, radius=0.007, shadow=False, zorder=3)
    draw_token_matrix(
        ax, s1_x + 0.189, p_card_y + 0.046, 0.123, 0.114,
        n_rows=6, n_cols=6,
        row_labels=["M1", "T790", "L792", "C797", "T854", "[PAD]"],
        palette=pal_prot, mask_rows=[5],
        title=r"Residue Tokens $X_P$",
        dim_label=r"$N_P \times 480$",
        zorder=4
    )

    # =========================================================================
    # --- STAGE 2: LATENT ALIGNMENT (x = 0.349 to 0.529) ---
    # =========================================================================
    s2_x, s2_w = 0.349, 0.180
    add_rounded_box(ax, s2_x, top_y0, s2_w, top_h,
                    facecolor="#FAFBFD", edgecolor="#D8E2EC", lw=1.3, radius=0.014, zorder=1)
    add_header_banner(ax, s2_x, banner_y, s2_w, banner_h,
                      "Stage 2  |  Alignment",
                      facecolor="#E6EDF5", textcolor="#2C3E50", edgecolor="#CBD5E1", fontsize=11.5)

    # Drug Linear Projection Box
    add_rounded_box(ax, s2_x + 0.010, d_card_y + 0.022, 0.106, 0.176,
                    facecolor="#FFFDF7", edgecolor="#F8C471", lw=1.15, radius=0.009, zorder=2)
    ax.text(s2_x + 0.063, d_card_y + 0.166, "Linear",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#9A6700", zorder=4)
    ax.text(s2_x + 0.063, d_card_y + 0.140, "Projection",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#9A6700", zorder=4)
    draw_vector_bar(ax, s2_x + 0.018, d_card_y + 0.082, 0.090, 0.028,
                    n_cells=6, colors=pal_drug, vertical=False, zorder=4)
    ax.text(s2_x + 0.063, d_card_y + 0.048, r"$512 \rightarrow 256$",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#935116", zorder=4)

    # Aligned Token Pillar E_L
    draw_vector_bar(ax, s2_x + 0.129, d_card_y + 0.044, 0.038, 0.132,
                    n_cells=7, colors=pal_drug, vertical=True,
                    label=r"$E_L$", sublabel=r"$N_L \times 256$", zorder=4)

    add_arrow(ax, s1_x + 0.316, d_card_y + 0.108, s2_x + 0.010, d_card_y + 0.108,
              color="#D68910", lw=1.6, mutation_scale=11)
    add_arrow(ax, s2_x + 0.116, d_card_y + 0.108, s2_x + 0.129, d_card_y + 0.108,
              color="#D68910", lw=1.5, mutation_scale=10)

    # Center IG Attribution Anchor Badge between E_L and E_P
    add_rounded_box(ax, s2_x + 0.010, 0.678, s2_w - 0.020, 0.036,
                    facecolor="#FDF2F2", edgecolor="#E67E73", lw=1.15, linestyle="--",
                    radius=0.006, shadow=False, zorder=5)
    ax.text(s2_x + s2_w / 2, 0.696,
            r"IG Anchor $(E_L, E_P)$",
            ha="center", va="center", fontsize=9.4, fontweight="bold", color="#A93226", zorder=6)

    # Protein Linear Projection Box
    add_rounded_box(ax, s2_x + 0.010, p_card_y + 0.016, 0.106, 0.176,
                    facecolor="#F4FBFD", edgecolor="#7FB3D5", lw=1.15, radius=0.009, zorder=2)
    ax.text(s2_x + 0.063, p_card_y + 0.160, "Linear",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#1F618D", zorder=4)
    ax.text(s2_x + 0.063, p_card_y + 0.134, "Projection",
            ha="center", va="center", fontsize=10.0, fontweight="bold", color="#1F618D", zorder=4)
    draw_vector_bar(ax, s2_x + 0.018, p_card_y + 0.076, 0.090, 0.028,
                    n_cells=6, colors=pal_prot, vertical=False, zorder=4)
    ax.text(s2_x + 0.063, p_card_y + 0.042, r"$480 \rightarrow 256$",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1F618D", zorder=4)

    # Aligned Token Pillar E_P
    draw_vector_bar(ax, s2_x + 0.129, p_card_y + 0.032, 0.038, 0.132,
                    n_cells=7, colors=pal_prot, vertical=True,
                    label=r"$E_P$", sublabel=r"$N_P \times 256$", zorder=4)

    add_arrow(ax, s1_x + 0.316, p_card_y + 0.098, s2_x + 0.010, p_card_y + 0.098,
              color="#5499C7", lw=1.6, mutation_scale=11)
    add_arrow(ax, s2_x + 0.116, p_card_y + 0.098, s2_x + 0.129, p_card_y + 0.098,
              color="#5499C7", lw=1.5, mutation_scale=10)

    # =========================================================================
    # --- STAGE 3: BILATERAL CROSS-ATTENTION & MAX POOLING (x = 0.540 to 0.988) ---
    # =========================================================================
    s3_x, s3_w = 0.540, 0.448
    add_rounded_box(ax, s3_x, top_y0, s3_w, top_h,
                    facecolor="#FAFBFD", edgecolor="#D8E2EC", lw=1.3, radius=0.014, zorder=1)
    add_header_banner(ax, s3_x, banner_y, s3_w, banner_h,
                      "Stage 3  |  Bilateral Cross-Attention & Masked Max Pooling",
                      facecolor="#E8EAF6", textcolor="#283593", edgecolor="#C5CAE9", fontsize=11.5)

    # 3A. D2T Cross-Attention Card (Top left of Stage 3)
    ca_x, ca_w = s3_x + 0.026, 0.266
    d2t_y, d2t_h = d_card_y + 0.006, 0.214
    add_rounded_box(ax, ca_x, d2t_y, ca_w, d2t_h,
                    facecolor="#FFFDF9", edgecolor="#F0B27A", lw=1.15, radius=0.010, zorder=2)
    add_rounded_box(ax, ca_x + 0.006, d2t_y + d2t_h - 0.038, ca_w - 0.012, 0.031,
                    facecolor="#FEF5E7", edgecolor="none", radius=0.005, shadow=False, zorder=3)
    ax.text(ca_x + ca_w / 2, d2t_y + d2t_h - 0.0225,
            r"D2T Cross-Attention ($\mathrm{Drug} \rightarrow \mathrm{Target}$)",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#935116", zorder=4)

    ax.text(ca_x + 0.014, d2t_y + 0.126, r"Query: $E_L$", ha="left", va="center",
            fontsize=9.6, fontweight="bold", color="#AF601A", zorder=4)
    ax.text(ca_x + 0.014, d2t_y + 0.072, r"Key/Val: $E_P$", ha="left", va="center",
            fontsize=9.6, fontweight="bold", color="#2471A3", zorder=4)

    # 4-Head MHA Block inside D2T
    add_rounded_box(ax, ca_x + 0.095, d2t_y + 0.026, 0.083, 0.132,
                    facecolor="#F5F7FF", edgecolor="#9FA8DA", lw=1.1, radius=0.007, shadow=False, zorder=3)
    ax.text(ca_x + 0.1365, d2t_y + 0.138, "4-Head MHA", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color="#3949AB", zorder=4)
    hm_cols = ["#FEF9E7", "#FAD7A0", "#F5B041", "#EC7063",
               "#FCF3CF", "#EB984E", "#E74C3C", "#FEF9E7",
               "#FDEBD0", "#F5B041", "#FEF9E7", "#EB984E",
               "#FAD7A0", "#EC7063", "#FCF3CF", "#F5B041"]
    for hr in range(4):
        for hc in range(4):
            rect = patches.Rectangle(
                (ca_x + 0.109 + hc * 0.0138, d2t_y + 0.038 + hr * 0.018),
                0.012, 0.015, facecolor=hm_cols[(hr * 4 + hc) % len(hm_cols)],
                edgecolor="#FFFFFF", lw=0.5, alpha=0.82, zorder=4
            )
            ax.add_patch(rect)

    # Arrow with residual (+) to E_hat_L
    add_arrow(ax, ca_x + 0.179, d2t_y + 0.092, ca_x + 0.197, d2t_y + 0.092,
              color="#7986CB", lw=1.5, mutation_scale=10)
    ax.text(ca_x + 0.188, d2t_y + 0.114, r"$\oplus$", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#5C6BC0", zorder=5)

    draw_token_matrix(
        ax, ca_x + 0.197, d2t_y + 0.042, 0.058, 0.102,
        n_rows=4, n_cols=4, row_labels=[],
        palette=pal_attn_l, mask_rows=[],
        title=r"$\hat{E}_L$", dim_label=r"$N_L \times 256$", zorder=4
    )

    # 3B. T2D Cross-Attention Card (Bottom left of Stage 3)
    t2d_y, t2d_h = p_card_y + 0.006, 0.214
    add_rounded_box(ax, ca_x, t2d_y, ca_w, t2d_h,
                    facecolor="#F7FBFE", edgecolor="#7FB3D5", lw=1.15, radius=0.010, zorder=2)
    add_rounded_box(ax, ca_x + 0.006, t2d_y + t2d_h - 0.038, ca_w - 0.012, 0.031,
                    facecolor="#EBF5FB", edgecolor="none", radius=0.005, shadow=False, zorder=3)
    ax.text(ca_x + ca_w / 2, t2d_y + t2d_h - 0.0225,
            r"T2D Cross-Attention ($\mathrm{Target} \rightarrow \mathrm{Drug}$)",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1F618D", zorder=4)

    ax.text(ca_x + 0.014, t2d_y + 0.126, r"Query: $E_P$", ha="left", va="center",
            fontsize=9.6, fontweight="bold", color="#2471A3", zorder=4)
    ax.text(ca_x + 0.014, t2d_y + 0.072, r"Key/Val: $E_L$", ha="left", va="center",
            fontsize=9.6, fontweight="bold", color="#AF601A", zorder=4)

    add_rounded_box(ax, ca_x + 0.095, t2d_y + 0.026, 0.083, 0.132,
                    facecolor="#F5F7FF", edgecolor="#9FA8DA", lw=1.1, radius=0.007, shadow=False, zorder=3)
    ax.text(ca_x + 0.1365, t2d_y + 0.138, "4-Head MHA", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color="#3949AB", zorder=4)
    hm_cols_p = ["#EBF5FB", "#85C1E9", "#5499C7", "#5C6BC0",
                 "#D6EAF8", "#5499C7", "#3949AB", "#EBF5FB",
                 "#AED6F1", "#2E86C1", "#EBF5FB", "#5499C7",
                 "#85C1E9", "#5C6BC0", "#D6EAF8", "#2E86C1"]
    for hr in range(4):
        for hc in range(4):
            rect = patches.Rectangle(
                (ca_x + 0.109 + hc * 0.0138, t2d_y + 0.038 + hr * 0.018),
                0.012, 0.015, facecolor=hm_cols_p[(hr * 4 + hc) % len(hm_cols_p)],
                edgecolor="#FFFFFF", lw=0.5, alpha=0.82, zorder=4
            )
            ax.add_patch(rect)

    add_arrow(ax, ca_x + 0.179, t2d_y + 0.092, ca_x + 0.197, t2d_y + 0.092,
              color="#7986CB", lw=1.5, mutation_scale=10)
    ax.text(ca_x + 0.188, t2d_y + 0.114, r"$\oplus$", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color="#5C6BC0", zorder=5)

    draw_token_matrix(
        ax, ca_x + 0.197, t2d_y + 0.042, 0.058, 0.102,
        n_rows=4, n_cols=4, row_labels=[],
        palette=pal_attn_p, mask_rows=[],
        title=r"$\hat{E}_P$", dim_label=r"$N_P \times 256$", zorder=4
    )

    # Cross-attention input arrows from E_L and E_P
    el_out_x = s2_x + 0.167
    ep_out_x = s2_x + 0.167
    add_arrow(ax, el_out_x, d2t_y + 0.126, ca_x, d2t_y + 0.126, color="#D68910", lw=1.6, mutation_scale=11)
    add_arrow(ax, ep_out_x, t2d_y + 0.126, ca_x, t2d_y + 0.126, color="#5499C7", lw=1.6, mutation_scale=11)
    add_arrow(ax, ep_out_x, p_card_y + 0.112, ca_x, d2t_y + 0.072,
              color="#5499C7", lw=1.4, linestyle="--", connectionstyle="arc3,rad=-0.12", mutation_scale=10)
    add_arrow(ax, el_out_x, d_card_y + 0.092, ca_x, t2d_y + 0.072,
              color="#D68910", lw=1.4, linestyle="--", connectionstyle="arc3,rad=0.12", mutation_scale=10)

    # 3C. Masked Adaptive Max Pooling & Complex Concatenation (Right side of Stage 3)
    pool_x, pool_w = ca_x + ca_w + 0.014, 0.132

    # Drug Masked Max Pool
    add_rounded_box(ax, pool_x, d2t_y + 0.056, pool_w, 0.150,
                    facecolor="#F4FBF8", edgecolor="#76D7C4", lw=1.15, radius=0.009, zorder=2)
    ax.text(pool_x + pool_w / 2, d2t_y + 0.176, "Masked Max Pool",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#117864", zorder=4)
    draw_vector_bar(ax, pool_x + 0.014, d2t_y + 0.114, pool_w - 0.028, 0.028,
                    n_cells=6, colors=pal_attn_l, vertical=False, zorder=4)
    ax.text(pool_x + pool_w / 2, d2t_y + 0.080, r"$v_L \in \mathbb{R}^{256}$",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#148F77", zorder=4)

    # Target Masked Max Pool
    add_rounded_box(ax, pool_x, t2d_y + 0.008, pool_w, 0.150,
                    facecolor="#F4FBF8", edgecolor="#76D7C4", lw=1.15, radius=0.009, zorder=2)
    ax.text(pool_x + pool_w / 2, t2d_y + 0.128, "Masked Max Pool",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#117864", zorder=4)
    draw_vector_bar(ax, pool_x + 0.014, t2d_y + 0.066, pool_w - 0.028, 0.028,
                    n_cells=6, colors=pal_attn_p, vertical=False, zorder=4)
    ax.text(pool_x + pool_w / 2, t2d_y + 0.032, r"$v_P \in \mathbb{R}^{256}$",
            ha="center", va="center", fontsize=10.2, fontweight="bold", color="#148F77", zorder=4)

    add_arrow(ax, ca_x + ca_w, d2t_y + 0.125, pool_x, d2t_y + 0.125, color="#48C9B0", lw=1.6, mutation_scale=11)
    add_arrow(ax, ca_x + ca_w, t2d_y + 0.088, pool_x, t2d_y + 0.088, color="#48C9B0", lw=1.6, mutation_scale=11)

    # Joint Complex Concatenation Box between v_L and v_P
    concat_y, concat_h = 0.654, 0.096
    add_rounded_box(ax, pool_x - 0.003, concat_y, pool_w + 0.006, concat_h,
                    facecolor="#FFFDF5", edgecolor="#E5B869", lw=1.2, radius=0.009, zorder=3)
    ax.text(pool_x + pool_w / 2, concat_y + 0.074, "Complex Fusion",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#7E5109", zorder=4)
    draw_vector_bar(ax, pool_x + 0.008, concat_y + 0.038, pool_w - 0.016, 0.024,
                    n_cells=10, colors=pal_attn_l[:5] + pal_attn_p[:5], vertical=False, zorder=4)
    ax.text(pool_x + pool_w / 2, concat_y + 0.016,
            r"$v_{\mathrm{complex}} = [v_L \| v_P]$",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#7E5109", zorder=4)

    add_arrow(ax, pool_x + pool_w / 2, d2t_y + 0.056, pool_x + pool_w / 2, concat_y + concat_h,
              color="#D4AC0D", lw=1.5, mutation_scale=10)
    add_arrow(ax, pool_x + pool_w / 2, t2d_y + 0.158, pool_x + pool_w / 2, concat_y,
              color="#D4AC0D", lw=1.5, mutation_scale=10)

    # =========================================================================
    # BOTTOM TIER: STAGE 4 (THERMODYNAMIC DUAL-BRANCH) & STAGE 5 (IG INTERPRETABILITY)
    # =========================================================================
    bot_y0, bot_h = 0.010, 0.424
    bot_banner_y = bot_y0 + bot_h - banner_h

    # --- STAGE 4: DUAL-BRANCH THERMODYNAMIC READOUT & REGULARIZATION (x = 0.008 to 0.542) ---
    s4_x, s4_w = 0.008, 0.534
    add_rounded_box(ax, s4_x, bot_y0, s4_w, bot_h,
                    facecolor="#FAFBFD", edgecolor="#D8E2EC", lw=1.3, radius=0.014, zorder=1)
    add_header_banner(ax, s4_x, bot_banner_y, s4_w, banner_h,
                      "Stage 4  |  Dual-Branch Thermodynamic Readout & Co-Training",
                      facecolor="#F9EBEA", textcolor="#78281F", edgecolor="#E6B0AA", fontsize=11.5)

    # Shared Complex Representation Card inside Stage 4
    vc_card_x, vc_card_y, vc_card_w, vc_card_h = s4_x + 0.012, bot_y0 + 0.030, 0.064, 0.324
    add_rounded_box(ax, vc_card_x, vc_card_y, vc_card_w, vc_card_h,
                    facecolor="#FFFDF5", edgecolor="#E5B869", lw=1.15, radius=0.008, shadow=False, zorder=3)
    vc_x, vc_y, vc_w, vc_h = vc_card_x + 0.012, vc_card_y + 0.036, 0.040, 0.232
    draw_vector_bar(ax, vc_x, vc_y, vc_w, vc_h,
                    n_cells=10, colors=pal_attn_l[:5] + pal_attn_p[:5], vertical=True,
                    label=r"$v_{\mathrm{complex}}$", sublabel=r"$\mathbb{R}^{512}$", zorder=4)

    # Clean routing bus from Stage 3 Complex Concatenation around the outer margin into Stage 4 v_complex card
    bus_y = 0.446
    vc_mid_y = vc_card_y + vc_card_h * 0.50
    ax.plot([pool_x + pool_w + 0.003, 0.994, 0.994, 0.003, 0.003],
            [concat_y + concat_h / 2, concat_y + concat_h / 2, bus_y, bus_y, vc_mid_y],
            color="#D4AC0D", lw=1.8, alpha=0.85, zorder=6)
    add_arrow(ax, 0.003, vc_mid_y, vc_card_x, vc_mid_y,
              color="#D4AC0D", lw=1.8, mutation_scale=11, zorder=6)

    # 4A. Primary Direct Regression Branch Card (Top of Stage 4)
    b1_x, b1_y, b1_w, b1_h = s4_x + 0.086, bot_y0 + 0.214, 0.260, 0.156
    add_rounded_box(ax, b1_x, b1_y, b1_w, b1_h,
                    facecolor="#F7FAFF", edgecolor="#93C5FD", lw=1.15, radius=0.009, zorder=2)
    ax.text(b1_x + 0.012, b1_y + b1_h - 0.022,
            r"1. Primary Regression Branch",
            ha="left", va="center", fontsize=10.0, fontweight="bold", color="#2B579A", zorder=4)

    last_m1 = draw_mlp_icon(ax, b1_x + 0.012, b1_y + 0.014, 0.122, 0.104,
                            layer_sizes=[4, 5, 4, 3, 1], r_y=0.0058,
                            node_colors=["#F8C471", "#85C1E9", "#9FA8DA", "#B39DDB", "#5499C7"],
                            zorder=4)

    # Output pill y_main
    add_rounded_box(ax, b1_x + 0.148, b1_y + 0.028, 0.102, 0.076,
                    facecolor="#EBF3FD", edgecolor="#7FB3D5", lw=1.15, radius=0.007, shadow=False, zorder=4)
    ax.text(b1_x + 0.199, b1_y + 0.078, "Affinity",
            ha="center", va="center", fontsize=9.4, fontweight="bold", color="#1F618D", zorder=5)
    ax.text(b1_x + 0.199, b1_y + 0.048, r"$y_{\mathrm{main}}$",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color="#2471A3", zorder=5)

    add_arrow(ax, last_m1[0] + 0.006, last_m1[1], b1_x + 0.148, b1_y + 0.066,
              color="#5499C7", lw=1.5, mutation_scale=10)

    # 4B. Auxiliary Thermodynamic Branch Card (Bottom of Stage 4)
    b2_x, b2_y, b2_w, b2_h = s4_x + 0.086, bot_y0 + 0.014, 0.260, 0.186
    add_rounded_box(ax, b2_x, b2_y, b2_w, b2_h,
                    facecolor="#FFF9FA", edgecolor="#F1948A", lw=1.15, radius=0.009, zorder=2)
    ax.text(b2_x + 0.012, b2_y + b2_h - 0.021,
            r"2. Thermodynamic Gibbs Branch",
            ha="left", va="center", fontsize=10.0, fontweight="bold", color="#922B21", zorder=4)

    last_h = draw_mlp_icon(ax, b2_x + 0.010, b2_y + 0.090, 0.070, 0.064,
                           layer_sizes=[3, 4, 1], r_y=0.0052,
                           node_colors=["#F8C471", "#F5B7B1", "#EC7063"], zorder=4)
    last_s = draw_mlp_icon(ax, b2_x + 0.010, b2_y + 0.014, 0.070, 0.064,
                           layer_sizes=[3, 4, 1], r_y=0.0052,
                           node_colors=["#F8C471", "#AED6F1", "#5499C7"], zorder=4)

    # Latent H Box
    add_rounded_box(ax, b2_x + 0.090, b2_y + 0.094, 0.074, 0.056,
                    facecolor="#FDEDEC", edgecolor="#EC7063", lw=1.1, radius=0.006, shadow=False, zorder=4)
    ax.text(b2_x + 0.127, b2_y + 0.122, r"Enthalpy $H$",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#922B21", zorder=5)

    # Latent S Box
    add_rounded_box(ax, b2_x + 0.090, b2_y + 0.018, 0.074, 0.056,
                    facecolor="#EBF5FB", edgecolor="#5499C7", lw=1.1, radius=0.006, shadow=False, zorder=4)
    ax.text(b2_x + 0.127, b2_y + 0.046, r"Entropy $S$",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1F618D", zorder=5)

    add_arrow(ax, last_h[0] + 0.005, last_h[1], b2_x + 0.090, b2_y + 0.122, color="#EC7063", lw=1.4, mutation_scale=9)
    add_arrow(ax, last_s[0] + 0.005, last_s[1], b2_x + 0.090, b2_y + 0.046, color="#5499C7", lw=1.4, mutation_scale=9)

    # Algebraic Difference Output y_phy = H - T*S
    add_rounded_box(ax, b2_x + 0.175, b2_y + 0.042, 0.077, 0.084,
                    facecolor="#FFFFFF", edgecolor="#D98880", lw=1.15, radius=0.006, shadow=False, zorder=4)
    ax.text(b2_x + 0.2135, b2_y + 0.100, "Gibbs Readout",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#7B241C", zorder=5)
    ax.text(b2_x + 0.2135, b2_y + 0.066, r"$y_{\mathrm{phy}} = H - T S$",
            ha="center", va="center", fontsize=9.0, fontweight="bold", color="#A93226", zorder=5)

    add_arrow(ax, b2_x + 0.164, b2_y + 0.122, b2_x + 0.175, b2_y + 0.095, color="#EC7063", lw=1.4, mutation_scale=9)
    add_arrow(ax, b2_x + 0.164, b2_y + 0.046, b2_x + 0.175, b2_y + 0.070, color="#5499C7", lw=1.4, mutation_scale=9)

    # Arrows from v_complex card into Branch 1 and Branch 2
    add_arrow(ax, vc_card_x + vc_card_w, vc_card_y + vc_card_h * 0.76, b1_x, b1_y + b1_h * 0.45,
              color="#5499C7", lw=1.6, mutation_scale=11)
    add_arrow(ax, vc_card_x + vc_card_w, vc_card_y + vc_card_h * 0.26, b2_x, b2_y + b2_h * 0.48,
              color="#EC7063", lw=1.6, mutation_scale=11)

    # 4C. Joint Co-Training Box (Right of Stage 4)
    loss_x, loss_y, loss_w, loss_h = s4_x + 0.358, bot_y0 + 0.014, 0.164, 0.356
    add_rounded_box(ax, loss_x, loss_y, loss_w, loss_h,
                    facecolor="#FFFDF7", edgecolor="#F0B27A", lw=1.15, radius=0.010, zorder=2)
    ax.text(loss_x + loss_w / 2, loss_y + loss_h - 0.025,
            "Joint Supervision", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#873600", zorder=4)

    add_rounded_box(ax, loss_x + 0.012, loss_y + 0.242, loss_w - 0.024, 0.056,
                    facecolor="#FFFFFF", edgecolor="#85C1E9", lw=1.1, radius=0.006, shadow=False, zorder=3)
    ax.text(loss_x + loss_w / 2, loss_y + 0.270, r"Main Loss $\mathcal{L}_{\mathrm{main}}$",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#1F618D", zorder=4)

    add_rounded_box(ax, loss_x + 0.008, loss_y + 0.126, loss_w - 0.016, 0.088,
                    facecolor="#FEF9E7", edgecolor="#F5B041", lw=1.2, radius=0.007, shadow=False, zorder=3)
    ax.text(loss_x + loss_w / 2, loss_y + 0.190, "Total Objective",
            ha="center", va="center", fontsize=9.4, fontweight="bold", color="#7E5109", zorder=4)
    ax.text(loss_x + loss_w / 2, loss_y + 0.154, r"$\mathcal{L} = \mathcal{L}_{\mathrm{main}} + \lambda_{\mathrm{phy}}\mathcal{L}_{\mathrm{phy}}$",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#9A6700", zorder=4)

    add_rounded_box(ax, loss_x + 0.012, loss_y + 0.042, loss_w - 0.024, 0.056,
                    facecolor="#FFFFFF", edgecolor="#F1948A", lw=1.1, radius=0.006, shadow=False, zorder=3)
    ax.text(loss_x + loss_w / 2, loss_y + 0.070, r"Gibbs Loss $\mathcal{L}_{\mathrm{phy}}$",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#922B21", zorder=4)

    add_arrow(ax, b1_x + b1_w, b1_y + 0.066, loss_x + 0.012, loss_y + 0.270,
              color="#5499C7", lw=1.5, mutation_scale=10)
    add_arrow(ax, b2_x + b2_w, b2_y + 0.084, loss_x + 0.012, loss_y + 0.070,
              color="#EC7063", lw=1.5, mutation_scale=10)
    add_arrow(ax, loss_x + loss_w / 2, loss_y + 0.242, loss_x + loss_w / 2, loss_y + 0.214,
              color="#D68910", lw=1.4, mutation_scale=9)
    add_arrow(ax, loss_x + loss_w / 2, loss_y + 0.098, loss_x + loss_w / 2, loss_y + 0.126,
              color="#D68910", lw=1.4, mutation_scale=9)

    # =========================================================================
    # --- STAGE 5: 64-STEP INTEGRATED GRADIENTS & INTERPRETABILITY (x = 0.552 to 0.988) ---
    # =========================================================================
    s5_x, s5_w = 0.552, 0.436
    add_rounded_box(ax, s5_x, bot_y0, s5_w, bot_h,
                    facecolor="#FAFBFD", edgecolor="#D8E2EC", lw=1.3, radius=0.014, zorder=1)
    add_header_banner(ax, s5_x, bot_banner_y, s5_w, banner_h,
                      "Stage 5  |  64-Step IG Attribution & Generalization",
                      facecolor="#E8F6F3", textcolor="#117864", edgecolor="#A3E4D7", fontsize=11.5)

    add_arrow(ax, loss_x + loss_w, loss_y + 0.170, s5_x + 0.012, bot_y0 + 0.190,
              color="#48C9B0", lw=1.6, linestyle="--", connectionstyle="arc3,rad=-0.06", mutation_scale=11)

    # Three Tall, Spacious Visual Interpretability Cards inside Stage 5
    c_y, c_h = bot_y0 + 0.014, 0.356
    c1_x, c1_w = s5_x + 0.012, 0.134
    c2_x, c2_w = c1_x + c1_w + 0.009, 0.144
    c3_x, c3_w = c2_x + c2_w + 0.009, 0.120

    # Card 1: 3D Pocket & Resistance Recovery
    add_rounded_box(ax, c1_x, c_y, c1_w, c_h,
                    facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.1, radius=0.009, zorder=2)
    ax.text(c1_x + c1_w / 2, c_y + c_h - 0.024, "3D Pocket Recovery",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#1E293B", zorder=4)

    crop_and_place_image(ax, pose_img,
                         extent=[c1_x + 0.006, c1_x + c1_w - 0.006, c_y + 0.056, c_y + c_h - 0.044],
                         crop_box=(0.05, 0.22, 0.95, 0.95), zorder=3, border_color="#E2E8F0")

    ax.text(c1_x + c1_w / 2, c_y + 0.028,
            "Binding Site & Resistance\nLoci (≤ 5 Å Contact)",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#922B21", zorder=4)

    # Card 2: 2D Enthalpy-Entropy Substructure Decoupling (Stacked Undistorted Landscape Layout)
    add_rounded_box(ax, c2_x, c_y, c2_w, c_h,
                    facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.1, radius=0.009, zorder=2)
    ax.text(c2_x + c2_w / 2, c_y + c_h - 0.024, "Enthalpy–Entropy 2D",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#1E293B", zorder=4)

    ig_h_img = os.path.join(FIG_DIR, "fig2_panel_c", "Afatinib_2_enthalpy_IG_H.png")
    ig_s_img = os.path.join(FIG_DIR, "fig2_panel_c", "Afatinib_3_entropy_IG_S.png")

    row_h = 0.114
    r1_y = c_y + 0.188
    r2_y = c_y + 0.062
    lbl_w = 0.042
    mol_x0 = c2_x + 0.006 + lbl_w
    mol_x1 = c2_x + c2_w - 0.006

    add_rounded_box(ax, c2_x + 0.005, r1_y, lbl_w - 0.002, row_h,
                    facecolor="#FFF5F5", edgecolor="#F5B7B1", lw=0.9, radius=0.004, shadow=False, zorder=3)
    ax.text(c2_x + 0.004 + lbl_w / 2, r1_y + 0.072, r"$|\mathrm{IG}(H)|$",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#A93226", zorder=4)
    ax.text(c2_x + 0.004 + lbl_w / 2, r1_y + 0.038, "Warhead",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#922B21", zorder=4)
    crop_and_place_image(ax, ig_h_img,
                         extent=[mol_x0, mol_x1, r1_y, r1_y + row_h],
                         crop_box=(0.03, 0.21, 0.96, 0.83), zorder=3, border_color="#F5B7B1")

    add_rounded_box(ax, c2_x + 0.005, r2_y, lbl_w - 0.002, row_h,
                    facecolor="#F4F9FD", edgecolor="#AED6F1", lw=0.9, radius=0.004, shadow=False, zorder=3)
    ax.text(c2_x + 0.004 + lbl_w / 2, r2_y + 0.072, r"$|\mathrm{IG}(S)|$",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#2471A3", zorder=4)
    ax.text(c2_x + 0.004 + lbl_w / 2, r2_y + 0.038, "Flex. Tail",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1F618D", zorder=4)
    crop_and_place_image(ax, ig_s_img,
                         extent=[mol_x0, mol_x1, r2_y, r2_y + row_h],
                         crop_box=(0.03, 0.21, 0.96, 0.83), zorder=3, border_color="#AED6F1")

    ax.text(c2_x + c2_w / 2, c_y + 0.028,
            "Orthogonal Pharmacophore\nAttribution",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#784212", zorder=4)

    # Card 3: OOD Generalization & Kinome Selectivity
    add_rounded_box(ax, c3_x, c_y, c3_w, c_h,
                    facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.1, radius=0.009, zorder=2)
    ax.text(c3_x + c3_w / 2, c_y + c_h - 0.024, "OOD & Selectivity",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#1E293B", zorder=4)

    regimes = [
        ("Warm-start", "#F2FBF7", "#76D7C4", "#117864"),
        ("Unseen drug", "#F4F9FD", "#85C1E9", "#1F618D"),
        ("Unseen target", "#FEF9E7", "#F8C471", "#9A6700"),
        ("All unseen", "#FDEDEC", "#F1948A", "#922B21"),
    ]
    for ri, (r_title, r_bg, r_ec, r_tc) in enumerate(regimes):
        ry = c_y + c_h - 0.098 - ri * 0.058
        add_rounded_box(ax, c3_x + 0.007, ry, c3_w - 0.014, 0.046,
                        facecolor=r_bg, edgecolor=r_ec, lw=1.05, radius=0.006, shadow=False, zorder=3)
        ax.text(c3_x + c3_w / 2, ry + 0.023, r_title,
                ha="center", va="center", fontsize=8.8, fontweight="bold", color=r_tc, zorder=4)

    ax.text(c3_x + c3_w / 2, c_y + 0.028,
            "Kinome-Wide\nSelectivity Profiling",
            ha="center", va="center", fontsize=8.4, fontweight="bold", color="#117864", zorder=4)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PNG, format="png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(ARTIFACT_PNG, format="png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Saved:", OUT_PDF, OUT_PNG, ARTIFACT_PNG)


if __name__ == "__main__":
    build_architecture_figure()
