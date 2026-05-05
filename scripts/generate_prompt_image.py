import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import os

# --- SETTINGS & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_prompt_image():
    # 1. Exact Physical Dimensions Requested (8.73 cm x 7.53 cm)
    width_in = 8.73 / 2.54
    height_in = 7.53 / 2.54

    # Ultra-high 600 DPI for crystal clear text when printed
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=600)

    # Remove margins so the final image is EXACTLY the requested size
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    # 2. Clean Academic Background (White card with subtle gray border)
    rect = patches.Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#cbd5e1",
                             linewidth=2.0, transform=ax.transAxes)
    ax.add_patch(rect)

    # 3. Professional IEEE Syntax Highlighting Colors
    BLUE = '#1e3a8a'
    RED = '#b91c1c'
    TEAL = '#0f766e'
    GRAY = '#4b5563'
    BLACK = '#0f172a'
    GREEN = '#15803d'
    YELLOW = '#b45309'

    # 4. Condensed Prompt Text (Pre-formatted to fit the 8.73cm width perfectly)
    render_data = [
        [("Code Listing 1: Enterprise CX Prompt Template", BLUE, "bold")],
        [("-" * 55, GRAY, "normal")],
        [("SYSTEM:", GRAY, "bold"), (" You are an elite CX Escalation Manager.", BLACK, "normal")],
        [("Model flagged this review as: ", BLACK, "normal"), ("{sentiment}", TEAL, "bold")],
        [("CUSTOMER REVIEW: ", BLACK, "normal"), ('"{review_text}"', TEAL, "normal")],
        [],
        [("DIRECTIVES (RULES OF ENGAGEMENT):", BLUE, "bold")],
        [("1. Compliance:", GRAY, "bold"), (" NEVER admit legal liability or make", BLACK, "normal")],
        [("   financial promises. Use policy-safe language.", BLACK, "normal")],
        [("2. Diagnosis:", GRAY, "bold"), (" Pinpoint the exact operational failure", BLACK, "normal")],
        [("   (e.g., Delivery Delay, Manufacturing Defect).", BLACK, "normal")],
        [("3. Tone Strategy:", GRAY, "bold")],
        [("   - ", GRAY, "normal"), ("Negative:", RED, "bold"), (" Empathetic, action-oriented.", BLACK, "normal")],
        [("   - ", GRAY, "normal"), ("Neutral:", YELLOW, "bold"), (" Inquisitive, seek feedback.", BLACK, "normal")],
        [("   - ", GRAY, "normal"), ("Positive:", GREEN, "bold"), (" Build brand loyalty.", BLACK, "normal")],
        [("4. Multilingual:", GRAY, "bold"), (" DRAFT must be written in the", BLACK, "normal")],
        [("   EXACT SAME LANGUAGE", BLUE, "bold"), (" as the original review.", BLACK, "normal")],
        [],
        [("OUTPUT FORMAT REQUIREMENTS:", BLUE, "bold")],
        [("SUMMARY:  ", GRAY, "bold"), ("[1-sentence executive summary]", GRAY, "normal")],
        [("WHY:      ", GRAY, "bold"), ("[Precise operational root cause]", GRAY, "normal")],
        [("SOLUTION: ", GRAY, "bold"), ("[Internal strategic action steps]", GRAY, "normal")],
        [("DRAFT:    ", GRAY, "bold"), ("[Professional customer-facing response]", GRAY, "normal")],
        [("-" * 55, GRAY, "normal")],
    ]

    # 5. Font and Spacing Configuration
    font_size = 6.2  # Mathematically sized for 8.73cm width
    line_height = 1.0 / 26.0  # Spacing out 24 lines evenly vertically
    start_y = 1.0 - (line_height * 1.5)
    start_x = 0.035  # Left margin padding

    fp_normal = FontProperties(family='monospace', size=font_size)
    fp_bold = FontProperties(family='monospace', size=font_size, weight='bold')

    y_ptr = start_y
    renderer = fig.canvas.get_renderer()

    # Dynamic Monospace Calibration (Ensures text never overlaps on any OS)
    calib_t = ax.text(0, 0, "X" * 50, fontproperties=fp_normal, transform=ax.transAxes)
    calib_bbox = calib_t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
    char_width = (calib_bbox.x1 - calib_bbox.x0) / 50.0
    calib_t.remove()

    # 6. Render the Text Line by Line
    for row in render_data:
        if not row:
            y_ptr -= line_height
            continue

        x_ptr = start_x
        for text, color, weight in row:
            fp = fp_bold if weight == "bold" else fp_normal
            ax.text(x_ptr, y_ptr, text, color=color, fontproperties=fp,
                    transform=ax.transAxes, va='top', ha='left')
            # Advance the X position cleanly based on text length
            x_ptr += len(text) * char_width

        y_ptr -= line_height

    # 7. Save Image exactly as constructed (no automatic cropping)
    output_path = os.path.join(OUTPUT_DIR, "11_prompt_template.png")
    plt.savefig(output_path, dpi=600, facecolor='#ffffff')
    plt.close()

    print(
        f"--- SUCCESS: Rendered ultra-crisp {width_in * 2.54:.2f}cm x {height_in * 2.54:.2f}cm image to {output_path} ---")


if __name__ == "__main__":
    generate_prompt_image()