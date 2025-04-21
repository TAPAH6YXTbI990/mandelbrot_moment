#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

const int width = 1200, height = 800;
const double xMin = -2.0, xMax = 1.0;
const double yMin = -1.5, yMax = 1.5;
const int maxIter = 1000;

Vec3b getColor(int iter) {
    if (iter == maxIter) {
        return Vec3b(0, 0, 0); 
    }

    double t = (double)iter / maxIter;
    int r = (int)(255 * abs(sin(0.3 * iter)));
    int g = (int)(255 * abs(sin(0.3 * iter + 0.7)));
    int b = (int)(255 * abs(sin(0.3 * iter + 1.4)));

    return Vec3b(b, g, r);
}

void generateFractal(Mat& image, int start_row, int rows_to_process) {
    for (int j = start_row; j < start_row + rows_to_process; ++j) {
        for (int i = 0; i < width; ++i) {
            double x0 = xMin + (xMax - xMin) * i / (width - 1);
            double y0 = yMin + (yMax - yMin) * j / (height - 1);
            double x = 0.0, y = 0.0;
            int iter = 0;

            while (x * x + y * y < 4 && iter < maxIter) {
                double x_temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = x_temp;
                ++iter;
            }

            image.at<Vec3b>(j - start_row, i) = getColor(iter);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    double start_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        Mat soloImg(height, width, CV_8UC3);
        auto start = chrono::high_resolution_clock::now();
        generateFractal(soloImg, 0, height);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "Sequential time: " << duration.count() << " sec" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const int localRows = height / size;
    const int localStart = rank * localRows;

    Mat localImg(localRows, width, CV_8UC3);

    auto start_mpi = chrono::high_resolution_clock::now();
    generateFractal(localImg, localStart, localRows);

    Mat result(height, width, CV_8UC3);
    
    if (rank == 0) {

        localImg.copyTo(result(Rect(0, localStart, width, localRows)));

        for (int i = 1; i < size; ++i) {
            int other_start = localRows * i;
            int other_rows = localRows;
           
            vector<uchar> buffer(other_rows * width * 3);
            MPI_Recv(buffer.data(), buffer.size(), MPI_UNSIGNED_CHAR,
                i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Mat temp(other_rows, width, CV_8UC3, buffer.data());
            temp.copyTo(result(Rect(0, other_start, width, other_rows)));
        }

        auto end_mpi = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_mpi - start_mpi;
        cout << "Parallel time: " << duration.count() << " sec" << endl;
    }
    else {
        MPI_Send(localImg.data, localRows * width * 3,
            MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    imshow("Mandelbrot", result);
    waitKey(0);

    MPI_Finalize();
    return 0;
}
