## Installation
### Prerequisites
- Makes sure that you have CMake installed
### For Windows
1. Clone the repository.
    ```
    git clone https://github.com/clarlzx/PathTracer.git
    ```

2. In the project folder, open a terminal and run the following lines:
    ```
    mkdir build
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64
    ```
    You can also use other IDE besides Visual Studio 2022.

3. In the **build** folder, locate the solution file and open it with the IDE that you specified in step 2. In the case of Visual Studio, 
    - In Solution Explorer right click on "PathTracer"
    - Click "Set as Startup Project"
    - Run :smile: 