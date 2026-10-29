import streamlit as st
import numpy as np
import proc_func
from PIL import Image

st.title('Image engine')

src_file = st.file_uploader(
    "Source image",
    type = ["jpg", "jpeg", "png"]
)

if src_file is not None:
    img = Image.open(src_file)
    img = np.array(img)
    st.image(img, caption = "Uploaded image", use_container_width = True)
    res_img = None
    st.subheader('Processing technique')
    operation = st.radio(
        'Choose an operation: ',
        ('Custom Convolution', 'Blurring', 'Edge Detection', 'Unsharp Masking'),
        horizontal = True
    )
