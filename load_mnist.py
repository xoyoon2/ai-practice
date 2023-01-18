import numpy as np
import struct as st

"""
Detailed description for converting MNIST data:
https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
About magic number:
https://en.wikipedia.org/wiki/File_format
"""
filename = {
    "train_image": "/content/drive/MyDrive/ai-practice/data/train-images.idx3-ubyte",
    "train_label": "/content/drive/MyDrive/ai-practice/data/train-labels.idx1-ubyte",
    "test_image": "/content/drive/MyDrive/ai-practice/data/t10k-images.idx3-ubyte",
    "test_label": "/content/drive/MyDrive/ai-practice/data/t10k-labels.idx1-ubyte",
}


def load_MNIST():
    train_imagefile = open(
        filename["train_image"], "rb"
    )  # readable binary mode, read() method returns bytes object

    train_imagefile.seek(0)  # file handle to 0
    magic = st.unpack(
        ">4B", train_imagefile.read(4)
    )  # b'\x00\x00\x08\x03' (bytes) to integer tuple
    num_train = st.unpack(">I", train_imagefile.read(4))[0]
    width = st.unpack(">I", train_imagefile.read(4))[0]
    height = st.unpack(">I", train_imagefile.read(4))[0]
    # train_X = np.zeros((num_train, width, height)) optional

    num_bytes = num_train * width * height
    train_X = np.asarray(
        st.unpack(">" + "B" * num_bytes, train_imagefile.read(num_bytes))
    ).reshape((num_train, width, height))

    train_imagefile.close()

    train_labelfile = open(filename["train_label"], "rb")

    train_labelfile.seek(0)
    magic = st.unpack(">4B", train_labelfile.read(4))
    if st.unpack(">I", train_labelfile.read(4))[0] != num_train:
        print("Alert: train images and labels don't match")
    train_Y = np.asarray(
        st.unpack(">" + "B" * num_train, train_labelfile.read(num_train))
    )

    train_labelfile.close()

    print("train_X:", train_X.shape, "train_Y:", train_Y.shape)

    test_imagefile = open(filename["test_image"], "rb")

    test_imagefile.seek(0)
    magic = st.unpack(">4B", test_imagefile.read(4))
    num_test = st.unpack(">I", test_imagefile.read(4))[0]
    width = st.unpack(">I", test_imagefile.read(4))[0]
    height = st.unpack(">I", test_imagefile.read(4))[0]
    # test_X = np.zeros((num_test, width, height)) optional

    num_bytes = num_test * width * height
    test_X = np.asarray(
        st.unpack(">" + "B" * num_bytes, test_imagefile.read(num_bytes))
    ).reshape((num_test, width, height))

    test_imagefile.close()

    test_labelfile = open(filename["test_label"], "rb")

    test_labelfile.seek(0)
    magic = st.unpack(">4B", test_labelfile.read(4))
    if st.unpack(">I", test_labelfile.read(4))[0] != num_test:
        print("Alert: test images and labels don't match")
    test_Y = np.asarray(st.unpack(">" + "B" * num_test, test_labelfile.read(num_test)))

    test_labelfile.close()

    print("test_X:", test_X.shape, "test_Y:", test_Y.shape)

    return train_X, train_Y, test_X, test_Y
