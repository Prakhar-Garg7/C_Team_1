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
    if operation == 'Custom Convolution':
        kernel_size = st.number_input(
            'Enter kernel size: ',
            value = 3,
            step = 2
        )
        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                kernel[i][j] = st.number_input(
                    f'Enter value ({i}, {j}): ',
                    value = 1 / (kernel_size * kernel_size),
                    step = 0.01,
                    format = '%.2f'
                )
        if st.button('Apply processing'):
            with st.spinner('Applying convolution...'):
                res_img = proc_func.convolve(img, kernel)
    elif operation == 'Blurring':
        type = st.radio(
            'Choose type: ',
            ('Median Blurring', 'Gaussian Blurring'),
            horizontal = True
        )

        if type == 'Median Blurring':
            kernel_size = st.number_input(
                'Enter kernel size: ',
                value = 5,
                step = 2
            )
            if st.button('Apply processing'):
                with st.spinner('Blurring image...'):
                    res_img = proc_func.median_blurring(img, kernel_size)
        else:
            kernel_size = st.number_input(
                'Enter kernel size: ',
                value = 9,
                step = 2
            )
            sigma = st.number_input(
                'Enter sigma: ',
                value = 1.4,
                step = 0.1,
                format = '%.2f'
            )
            if st.button('Apply processing'):
                with st.spinner('Blurring image...'):
                    res_img = proc_func.gaussian_blurring(img, kernel_size, sigma)
    elif operation == 'Edge Detection':
        if st.button('Apply processing'):
            with st.spinner('Detecting edges...'):
                res_img = proc_func.laplacian_edge_detection(img)

    else:
        amp_factor = st.number_input(
            'Enter sharpening factor: ',
            value = 2,
            step = 1
        )
        if st.button('Apply processing'):
            with st.spinner('Sharpening edges...'):
                res_img = proc_func.unsharp_masking(img, amp_factor)

    if res_img is not None:
        st.subheader('Resulting image')
        st.image(res_img, caption = 'Resultant image', width = 'stretch')

