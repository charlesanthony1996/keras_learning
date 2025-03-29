image = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 8, 7, 6],
    [5, 4, 3, 2]
]

kernel =[
    [1, 0],
    [0, -1]
]
print("length of the kernel: ", len(kernel))

def convolve(image, kernel):
    kernel_size = len(kernel)
    output = []
    for i in range(len(image) - kernel_size + 1):
        row = []
        for j in range(len(image[0]) - kernel_size + 1):
            val = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    val += image[i+ ki][j + kj] * kernel[ki][kj]
            
            row.append(val)
        output.append(row)
    
    return output


conv_result = convolve(image, kernel)
print(conv_result)

# max pooling