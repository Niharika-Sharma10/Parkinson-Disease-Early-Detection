import streamlit as st
from PIL import Image
import tempfile

def upload_images():

    st.subheader("Upload Required Images")

    spiral_file = st.file_uploader(
        "Upload Spiral Drawing",
        type=["jpg", "png", "jpeg"]
    )

    wave_file = st.file_uploader(
        "Upload Wave Drawing",
        type=["jpg", "png", "jpeg"]
    )

    spiral_path = None
    wave_path = None

    # -------- Spiral Image --------
    if spiral_file is not None:
        try:
            image1 = Image.open(spiral_file)

            st.image(
                image1,
                caption="Spiral Preview",
                use_container_width=True
            )

            # Save temporary file
            temp_spiral = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image1.save(temp_spiral.name)

            spiral_path = temp_spiral.name

        except Exception:
            st.error("Invalid Spiral Image")

    # -------- Wave Image --------
    if wave_file is not None:
        try:
            image2 = Image.open(wave_file)

            st.image(
                image2,
                caption="Wave Preview",
                use_container_width=True
            )

            # Save temporary file
            temp_wave = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image2.save(temp_wave.name)

            wave_path = temp_wave.name

        except Exception:
            st.error("Invalid Wave Image")

    return spiral_path, wave_path