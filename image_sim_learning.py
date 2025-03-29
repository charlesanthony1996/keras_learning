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

# max pooling (2 x 2 window, stride 2)
def max_pool(matrix, pool_size=2, stride=2):
    output = []
    for i in range(0, len(matrix) - pool_size + 1, stride):
        row = []
        for j in range(0, len(matrix[0]) - pool_size + 1, stride):
            window = [matrix[i+x][j+y] for x in range(pool_size) for y in range(pool_size)]
            row.append(max(window))
        output.append(row)
    return output

pooled = max_pool(conv_result)
print(pooled)

# flatten

def flatten(matrix):
    return [value for row in matrix for value in row]

flat = flatten(pooled)
print("flattened: ", flat)

# fully connected (1 inputs -> 2 outputs)
def dense_layer(inputs, weights, biases):
    output = []
    for i in range(len(weights)):
        total = sum([inputs[j] * weights[i][j] for j in range(len(inputs))]) + biases[i]
        output.append(total)
    
# lets say we want 2 outputs
weights = [[0.5], [-0.8]]
biases = [1.0, 0.0]

fc_output = dense_layer(flat, weights, biases)
print("fully connected output: ", fc_output)