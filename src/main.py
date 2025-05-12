import random
import heapq
import matplotlib.pyplot as plt
import numpy as np
import imageio
import webbrowser

# Directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def generate_maze(width, height, seed = None):
    if seed is not None:
        random.seed(seed)
    
    maze = [[1 for i in range(width)] for j in range(height)]

    def make_passages(cx, cy):
        dirs = DIRECTIONS.copy()
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx + dx * 2, cy + dy * 2
            if 0 < nx < width and 0 < ny < height and maze[ny][nx] == 1:
                maze[cy + dy][cx + dx] = 0
                maze[ny][nx] = 0
                make_passages(nx, ny)

    maze[1][1] = 0
    make_passages(1, 1)
    return maze


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan Distance


def a_star_search(maze, start, goal):
    height = len(maze)
    width = len(maze[0])
    
    open_list = []
    close_list = set()
    cost = {start: 0}
    
    heapq.heappush(open_list, (0, start))
    
    came_from = {}
    frames = []

    def save_frame(current_path=None):
        frame = np.array(maze, dtype=float)
        for y, x in close_list:
            frame[y][x] = 0.7  # close_list
        if current_path:
            for y, x in current_path:
                frame[y][x] = 0.5  # Current path
        frame[start[0]][start[1]] = 0.3
        frame[goal[0]][goal[1]] = 0.9
        frames.append(np.copy(frame))

    while open_list:
        n, current = heapq.heappop(open_list)
        close_list.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path = path[::-1]
            for i in range(len(path)):
                save_frame(path[: i + 1])
            return frames

        for dy, dx in DIRECTIONS:
            ny, nx = current[0] + dy, current[1] + dx
            adjacent = (ny, nx)
            if 0 <= ny < height and 0 <= nx < width and maze[ny][nx] == 0:
                next_cost = cost[current] + 1 # since everytime we move one cell
                if adjacent not in cost or next_cost < cost[adjacent]:
                    came_from[adjacent] = current
                    cost[adjacent] = next_cost
                    f_c = next_cost + heuristic(goal, adjacent)
                    heapq.heappush(open_list, (f_c, adjacent))
        save_frame()

    return frames


def save_gif(frames, filename='a_star_maze.gif'):
    images = []
    for frame in frames:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame, cmap='Blues')
        ax.axis('off')
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8') # 1D numpy array of bytes. (8-bit unsigned int)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,)) # every pixel consists of 4 values. (rgba)
        images.append(image)
        plt.close(fig)

    imageio.mimsave(filename, images, fps=60)
    print("GIF saved as", filename)


# Parameters
def make_odd(n):
    return n if n % 2 == 1 else n + 1


def main():
    # width, height = input("Enter width: "), input("Enter height: ")

    # if(width.isnumeric() == False and height.isnumeric() == False):
    #     print("Invalid Inputs!")
    #     return

    # width, height = make_odd(int(width)), make_odd(int(height))
    
    width, height = make_odd(31), make_odd(31)
    
    start, goal =  (1, 1), (height - 2, width - 2)
    
    seed = width*height
    maze = generate_maze(width, height)
    
    # for i in maze:
    #     print(i)
    
    frames = a_star_search(maze, start, goal)
    print("Generating...")
    save_gif(frames)
    
    gif = r"vscode://file/C:/Users/liban/OneDrive/Desktop/MazeSolverGIF/a_star_maze.gif"
    webbrowser.open(gif)
    
    
if __name__ == "__main__":
    main()
else:
    print("Process Terminated!")