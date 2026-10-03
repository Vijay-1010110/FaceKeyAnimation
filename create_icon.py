import os
from PIL import Image, ImageDraw

def create_facekey_icon():
    os.makedirs("assets", exist_ok=True)
    size = (256, 256)
    
    # Create RGBA image
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Background rounded rectangle with sleek dark tech gradient
    bg_color = (15, 23, 42, 255) # Slate 900
    border_color = (56, 189, 248, 255) # Cyan 400
    
    # Draw rounded background
    draw.rounded_rectangle([(8, 8), (248, 248)], radius=52, fill=bg_color, outline=border_color, width=6)

    # Subtle inner glow / second border
    draw.rounded_rectangle([(16, 16), (240, 240)], radius=44, outline=(30, 41, 59, 255), width=3)

    # 2. Stylized Face Contour (Oval)
    face_contour = [
        (128, 42), (170, 56), (198, 92), (208, 138), (196, 184), 
        (168, 218), (128, 232), (88, 218), (60, 184), (48, 138), 
        (58, 92), (86, 56)
    ]
    # Draw face polygon outline
    draw.polygon(face_contour, outline=(14, 165, 233, 180), width=3)

    # 3. Eyebrows & Eyes
    # Left eye
    draw.arc([(72, 102), (108, 126)], start=0, end=360, fill=(56, 189, 248, 255), width=3)
    draw.ellipse([(86, 110), (94, 118)], fill=(240, 249, 255, 255))
    # Right eye
    draw.arc([(148, 102), (184, 126)], start=0, end=360, fill=(56, 189, 248, 255), width=3)
    draw.ellipse([(162, 110), (170, 118)], fill=(240, 249, 255, 255))

    # Eyebrows
    draw.line([(68, 92), (90, 84), (112, 90)], fill=(56, 189, 248, 255), width=3)
    draw.line([(144, 90), (166, 84), (188, 92)], fill=(56, 189, 248, 255), width=3)

    # 4. Nose Bridge & Tip
    draw.line([(128, 88), (128, 142)], fill=(56, 189, 248, 200), width=3)
    draw.line([(120, 150), (128, 154), (136, 150)], fill=(56, 189, 248, 255), width=3)

    # 5. Lips & Mouth Motion Track
    lip_outer = [(100, 178), (128, 172), (156, 178), (142, 196), (128, 198), (114, 196)]
    draw.polygon(lip_outer, outline=(244, 63, 94, 255), width=4) # Rose 500
    # Center mouth line
    draw.line([(106, 184), (128, 182), (150, 184)], fill=(251, 113, 133, 255), width=3)

    # 6. Facial Landmark Nodes (Bright Cyberpunk glowing cyan dots)
    nodes = [
        (128, 42), (170, 56), (198, 92), (208, 138), (196, 184), 
        (168, 218), (128, 232), (88, 218), (60, 184), (48, 138), 
        (58, 92), (86, 56), (90, 114), (166, 114), (128, 154),
        (100, 178), (156, 178), (128, 172), (128, 198)
    ]
    for nx, ny in nodes:
        draw.ellipse([(nx-4, ny-4), (nx+4, ny+4)], fill=(240, 249, 255, 255), outline=(14, 165, 233, 255), width=1)

    # 7. Recording Indicator (Glowing Red/Orange Studio Live badge at top right)
    draw.ellipse([(200, 24), (226, 50)], fill=(239, 68, 68, 255), outline=(254, 202, 202, 255), width=2)
    draw.ellipse([(208, 32), (218, 42)], fill=(255, 255, 255, 255))

    # Save PNG and multi-resolution Windows ICO
    png_path = os.path.abspath("assets/facekey_icon.png")
    ico_path = os.path.abspath("assets/facekey_icon.ico")
    
    img.save(png_path, format="PNG")
    
    # Generate multi-size ICO
    icon_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    img.save(ico_path, format="ICO", sizes=icon_sizes)
    print(f"[OK] Generated icon at: {ico_path}")
    print(f"[OK] Generated PNG at: {png_path}")

if __name__ == "__main__":
    create_facekey_icon()
