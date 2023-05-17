#ifndef CUDACC_RTC
#define CUDACC_RTC
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "device_functions.h"
#include <opencv2/opencv.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include<assert.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <curand.h>
#include<curand_kernel.h>
#include<vector>
#include<numeric>

using namespace std;
using namespace std::chrono;
#define AMOUNT 100000000
#define GRID_SIZE 128

using namespace cv;
using namespace std;

__global__ void thresholdImageKernel(unsigned char* colorData, unsigned char* thresholdData,
	int colorPitch, int thresholdPitch, int rows, int cols) {

	// Вычисляем номер текущего столбца и строки на основе индексов блока и нити
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Проверяем, что мы находимся в пределах изображения

	if (row < rows && col < cols) {

		// Определяем указатель на начало строки из colorData, на которой мы находимся

		unsigned char* rowPtr = colorData + row * colorPitch;

		// Получаем значение цвета текущего пикселя

		unsigned char color = rowPtr[col * 3];

		// Если значение цвета равно 255, то значение порога устанавливаем в 255, в противном случае в 0

		thresholdData[row * thresholdPitch + col] = (color == 255) ? 255 : 0;
	}
}



__global__ void HoughTransformKernel(unsigned char* src, size_t rows, size_t cols, size_t pitch, int* accumulator,
	float diagonal)
{	
	// Вычисляем глобальные индексы элемента изображения, на который мы ссылаемся
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// Проверяем, что мы находимся внутри границ изображения
	if (row < rows && col < cols) {
		// Проверяем, что текущий пиксель белый (имеет значение больше 0)
		if (*(src + pitch * row + col) > 0)
		{
			// Проходим по всем возможным углам (от 0 до 179 градусов)
			for (int t = 0; t < 180; t++)
			{
				// Вычисляем значение радиуса r для текущего угла и пикселей x, y
				float r = (float)(col * cosf(t * CV_PI / 180) + row * sinf(t * CV_PI / 180));

				// Вычисляем индекс массива для данного значения r и сохраняем его в irho
				int irho = int(r + diagonal / 2);

				// Добавляем единицу к соответствующему элементу массива накопителей accumulator
				atomicAdd(accumulator + 180 * irho + t, 1);
			}
		}
	}
}

//Это объявление функции ядра CUDA. Она используется для поворота изображения на заданный угол вокруг заданной точки. Аргументы функции включают 
//указатели на входное и выходное изображение, размеры изображения, угол поворота и точку центра.
__global__ void rotateImageKernel(uchar* src, uchar* dst, int srcPitch, int dstPitch, int cols, int rows, double angle, int channels,
	int centerX, int centerY, double sinAngle, double cosAngle)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;  //вычисление глобальных координат текущего пикселя на основе блочных и нитевых индексов.
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < rows && x < cols) //находится ли точка внутри изображения
	{
		double xRotated = (x - centerX) * cosAngle - (y - centerY) * sinAngle + centerX;  //выполняют преобразование координат для поворота пикселя на заданный угол вокруг заданной точки.
		double yRotated = (x - centerX) * sinAngle + (y - centerY) * cosAngle + centerY;

		//Определение целочисленных координат x1, x2, y1, y2 через функции floor и ceil для повернутого пикселя (xRotated, yRotated).

		int x1 = floor(xRotated);   //floor(x) возвращает наибольшее целое число, которое не превышает x. 
		int x2 = ceil(xRotated);    //ceil(x) возвращает наименьшее целое число, которое не меньше x
		int y1 = floor(yRotated);
		int y2 = ceil(yRotated);

		//Вычисление дробных значений dx1, dx2, dy1, dy2 для интерполяции между четырьмя ближайшими пикселями.

		double dx1 = xRotated - x1;
		double dx2 = x2 - xRotated;
		double dy1 = yRotated - y1;
		double dy2 = y2 - yRotated;

		//Эти четыре строки гарантируют, что координаты соседних пикселей находятся внутри границ изображения.

		x1 = max(0, min(x1, cols - 1));
		x2 = max(0, min(x2, cols - 1));
		y1 = max(0, min(y1, rows - 1));
		y2 = max(0, min(y2, rows - 1));

		for (int c = 0; c < channels; c++)
		{
			//Для каждого пикселя изображения (x, y) функция вычисляет значение (value) в точке, повернутой на угол angle относительно точки (centerX, centerY). 
			//Это значение вычисляется как сумма взвешенных значений 4-х ближайших пикселей. 
			//Результат записывается в соответствующий пиксель нового изображения (dst).
			double value = src[y1 * srcPitch + x1 * channels + c] * dx2 * dy2           
				+ src[y1 * srcPitch + x2 * channels + c] * dx1 * dy2
				+ src[y2 * srcPitch + x1 * channels + c] * dx2 * dy1
				+ src[y2 * srcPitch + x2 * channels + c] * dx1 * dy1;
			dst[y * dstPitch + x * channels + c] = static_cast<uchar>(value);
		}
	}
}


int main(int argc, char** argv)
{
	// Загрузка изображения с диска в переменную h_src с помощью функции imread()
	Mat source_img = cv::imread("C://Users/user/Pictures/car.jpg", cv::IMREAD_COLOR);

	// Генерация случайного угла поворота в радианах в диапазоне [-π, π]
	float random_corner = (float)(rand() % 360 - 180) * CV_PI / 180.0;

	// Определение координат начальной точки линии, проходящей через центр изображения
	float x_start = source_img.cols / 2;
	float y_start = source_img.rows / 2;

	// Рисование линии на изображении h_src, проходящей через точку (x_start, y_start) и направленной под углом angle
	line(source_img,
		Point(x_start, y_start),
		Point(x_start + 500 * cos(random_corner), y_start + 500 * sin(random_corner)),
		// Цвет линии (синий)
		Scalar(255, 0, 0),
		6);// Толщина линии

	// Отображение исходного изображения h_src на экране
	imshow("source", source_img);

	// Ожидание нажатия клавиши пользователем
	waitKey(0);

	// Получаем указатель на массив значений цветов пикселей изображения
	unsigned char* colorData = source_img.ptr<unsigned char>();
	size_t rows = source_img.rows; // количество строк в изображении
	size_t cols = source_img.cols; // количество столбцов в изображении


	// Выделяем память на GPU для хранения массива значений цветов пикселей
	uchar* d_colorData;
	// Выделяем память на GPU для хранения массива пороговых значений пикселей
	uchar* d_thresholdData;
	// Вычисляем количество байт, необходимых для хранения каждой строки изображения на GPU
	size_t colorPitch;
	// Вычисляем количество байт, необходимых для хранения каждой строки порогового изображения на GPU
	size_t threshholdPitch;

	// Выделяем память на GPU с учетом "выравнивания" по границам
	cudaMallocPitch((void**)&d_colorData, &colorPitch, cols * 3 * sizeof(unsigned char), rows);

	
	// Выделяем память на GPU с учетом "выравнивания" по границам
	cudaMallocPitch((void**)&d_thresholdData, &threshholdPitch, cols * sizeof(unsigned char), rows);

	// Копируем значения цветов пикселей с CPU на GPU
	cudaMemcpy2D(d_colorData, colorPitch, colorData, cols * 3 *
		sizeof(unsigned char),
		cols * 3 * sizeof(unsigned char), rows, cudaMemcpyHostToDevice);

	// Устанавливаем размеры блока и сетки
	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	// Вызываем ядро для выполнения пороговой обработки изображения
	thresholdImageKernel << <grid, block >> > (d_colorData, d_thresholdData, colorPitch,
		threshholdPitch, rows, cols);


	// вычисление диагонали изображения
	float diagonal = sqrt((float)cols * cols + (float)rows * rows);

	// выделение памяти на устройстве для массива аккумуляторов
	int* d_accumulator;
	cudaMalloc((void**)&d_accumulator, 180 * (int)diagonal * sizeof(int));

	// запуск ядра HoughTransformKernel на устройстве для заполнения массива аккумуляторов
	HoughTransformKernel << <grid, block >> > (d_thresholdData, rows, cols, threshholdPitch, d_accumulator, diagonal);

	// копирование массива аккумуляторов с устройства на хост
	int* accumulator = new int[180 * (int)diagonal];
	cudaMemcpy(accumulator, d_accumulator, 180 * (int)diagonal * sizeof(int), cudaMemcpyDeviceToHost);

	// создание вектора для хранения найденных линий
	std::vector<cv::Vec2f> lines;

	// проход по массиву аккумуляторов
	for (int r = 0; r < (int)diagonal; r++)
	{
		for (int t = 0; t < 180; t++)
		{
			// если значение в аккумуляторе больше или равно 500, то это возможная линия
			if (accumulator[180 * r + t] >= 500)
			{
				// создание вектора для хранения параметров найденной линии
				cv::Vec2f line(r - (int)diagonal / 2, t);
				// добавление линии в вектор
				lines.push_back(line);
				// если найдено достаточно линий, то завершить поиск
				if (lines.size() >= 20)
				{
					break;
				}
			}
		}
	}

	// вывод найденных линий на экран
	//for (auto item : lines)
	//{
	//	std::cout << item << std::endl;
	//}

	cout << endl << endl ;
	
	// Выделяем память на устройстве для хранения результата
	uchar* d_result;
	size_t resultPitch;
	// Вычисляем sin и cos угла поворота
	double sinAngle = sin(lines[0][1] * CV_PI / 180);
	double cosAngle = cos(lines[0][1] * CV_PI / 180);
	// Выделяем память на устройстве для хранения результата поворота
	cudaMallocPitch((void**)&d_result, &resultPitch, cols * 3 * sizeof(unsigned char), rows);
	// Запускаем ядро для поворота изображения
	rotateImageKernel << <grid, block >> > (d_colorData, d_result, colorPitch, resultPitch, cols, rows, lines[0][1] * CV_PI / 180, 3,
		cols / 2, rows / 2, sinAngle, cosAngle);
	// Выделяем память на хосте для хранения результата
	uchar* result = new unsigned char[rows * cols * 3];
	// Копируем результат с устройства на хост
	cudaMemcpy2D(result, cols * 3 * sizeof(uchar), d_result, resultPitch,
		cols * 3 * sizeof(unsigned char), rows, cudaMemcpyDeviceToHost);
	// Создаем объект типа Mat из результата
	Mat cv_dst(rows, cols, CV_8UC3, result);
	// Выводим результат
	imshow("result", cv_dst);
	waitKey(0);
	destroyAllWindows();
	return 0;
}