// avp2.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <cstdlib>
#include <malloc.h>
#include <iostream>
#include <iomanip>
#include <future>
#include <windows.h>

using namespace std;

const long long int MB = 1024 * 1024; // Константа, определяющая мегабайт
const long long int CACHE_LINE_SIZE_BYTES = 64; // Размер кэш-линии
const long long int CACHE_L3_SIZE_BYTES = 8 * MB; // Размер кэша L3
const long long int OFFSET_BYTES = 88 * MB; // Размер смещения
const long long int OFFSET_SIZE_INT = OFFSET_BYTES / sizeof(long long int); // Количество элементов, которые помещаются в смещение
const long long int MAX_ASSOCIATION = 20; // Максимальная степень ассоциативности


void initialize(long long int* arr, const long long int associativity, const long long int elements_in_block)
{
	for (long long int element_index = 0; element_index < elements_in_block; element_index++)
	{
		// Инициализация каждого элемента блока
		for (long long int block_index = 0; block_index < associativity - 1; block_index++)
		{
			// Заполнение элементов блока значениями, соответствующими его индексу и смещению
			arr[block_index * OFFSET_SIZE_INT + element_index] = (block_index + 1) * OFFSET_SIZE_INT + element_index;
		}
		if (element_index == elements_in_block - 1)
		{
			// Установка последнего элемента блока в 0
			arr[(associativity - 1) * OFFSET_SIZE_INT + element_index] = 0;
		}
		else
		{
			// Заполнение элемента блока значением, соответствующим следующему элементу в блоке
			arr[(associativity - 1) * OFFSET_SIZE_INT + element_index] = element_index + 1;
		}
	}
}
unsigned long long int read_array_through(long long int array[])
{
	// Индекс элемента массива
	long long int index = 0;
	// Количество повторов чтения элементов массива
	const int tries = 200;

	// Запоминаем время начала измерений в тактах процессора
	const long long int start_time = __rdtsc();

	// Цикл чтения элементов массива
	for (int i = 0; i < tries; i++)
	{
		// Используем каждый элемент массива в качестве индекса для следующего элемента
		do
		{
			index = array[index];
		} while (index != 0);
	}

	// Запоминаем время окончания измерений в тактах процессора
	const long long int end_time = __rdtsc();

	// Вычисляем количество тактов, затраченных на чтение массива
	return (end_time - start_time) / tries;
}

void plot( long long int y[], int n) {
	const int width = 80; // Ширина окна
	const int height = 24; // Высота окна
	const int xOffset = 2; // Смещение по x
	const int yOffset = 1; // Смещение по y

	COORD coord = { 0, 0 }; // Начальная позиция курсора 
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE); // Получение дескриптора консоли 

	// Ось X и метки
	COORD pos = { xOffset, height - yOffset };
	SetConsoleCursorPosition(hConsole, pos); // Перемещение курсора в позицию на экране

	std::cout << "\n 0 --------------------------------------------------------------------------------------------------->"; // Отображение оси X на экране
	for (int i = 1; i < width - xOffset; i++) { // Цикл по ширине окна
		pos.X = i + xOffset;
		SetConsoleCursorPosition(hConsole, pos); // Перемещение курсора в новую позицию

		if (i % 10 == 0) { // Если координата кратна 10
			pos.Y = height - yOffset + 1; // Перемещение курсора ниже оси X
			SetConsoleCursorPosition(hConsole, pos); // Перемещение курсора в новую позицию
		}
	}

	// Ось Y и метки
	for (int i = 0; i < height; i++) { // Цикл по высоте окна
		pos.X = xOffset - 1;
		pos.Y = height - yOffset - i;
		SetConsoleCursorPosition(hConsole, pos); // Перемещение курсора в новую позицию

		if (i == 23) std::cout << "^"; // Первая метка на оси Y
		else
			std::cout << "|"; // Остальные метки на оси Y

		if (i % 5 == 0) { // Если координата кратна 5
			pos.X = xOffset - 3; // Перемещение курсора влево от оси Y
			SetConsoleCursorPosition(hConsole, pos); // Перемещение курсора в новую позицию
		}
	}

	// График
	for (int i = 0; i < n; i++) { // Цикл по количеству точек графика
		int px = i*4; // Вычисление координаты x в пикселях
		int py = (int)((y[i] - y[0]) * (height - yOffset * 2) / (y[n - 1] - y[0])); // Вычисление координаты y в пикселях
		pos.X = px + xOffset; // Перемещение курсора по оси X в соответствующее место на графике
		pos.Y = height - py - yOffset; // Перемещение курс
		SetConsoleCursorPosition(hConsole, pos);
		std::cout << "*"; // Отображение точки графика
	}

	std::cin.get(); // Ожидание нажатия клавиши
}

int main()
{
	const int CACHE_SIZE = OFFSET_BYTES * MAX_ASSOCIATION; //Вычисление общего размера кэш-памяти.
	const int COUNT_OF_BLOCKS = CACHE_SIZE / sizeof(long long int); //Вычисление количества блоков в кэше.

	long long int times[20]; //Инициализация массива для хранения времени доступа для каждой степени ассоциативности.

	for (auto associativity = 1; associativity < MAX_ASSOCIATION + 1; associativity++) //Цикл по всем степеням ассоциативности кэша (от 1 до 20).
	{
		long long int* _array = static_cast<long long int*>(_aligned_malloc(CACHE_SIZE, CACHE_LINE_SIZE_BYTES)); //Выделение выровненной памяти размера CACHE_SIZE для массива.

		const long long int elements_in_block = ceil((double)CACHE_L3_SIZE_BYTES / (sizeof(long long int) * (associativity))); //вычисление количество элементов (long long int), которые могут быть хранены в каждом кэш-блоке.

		initialize(_array, associativity, elements_in_block); //Инициализация массива.

		unsigned long long int time = read_array_through(_array); //Вычисление времени доступа к кэш-памяти для текущей степени ассоциативности.

		_aligned_free(_array); //Освобождение выделенной памяти.

		cout << left << setw(5) << setfill(' ') << associativity << time << " clock cycles" << endl; //Вывод на экран времени доступа к кэш-памяти для текущей степени ассоциативности.

		times[associativity - 1] = time; //Сохранение времени доступа в массив.

	}

	system("pause"); //Ожидание нажатия клавиши пользователем.
	system("cls"); //Очистка консоли.

	plot(times, 20); //Построение графика на основе времени доступа к кэш-памяти.

	return 0; //Возвращение значения 0 (успешное завершение программы).
}
