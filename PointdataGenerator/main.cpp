#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cmath>
#include <sstream>
#include <iomanip>

int GRID_SIZE = 256 + 1;

struct Point
{
	Point()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	Point(float a, float b, float c)
	{
		x = a;
		y = b;
		z = c;
	}
	float x;
	float y;
	float z;
};

std::vector<std::vector<Point>> diamondSquare(float startDeviation);
void diamondStep(int x, int y, int stepSize, float roughness, std::vector<std::vector<Point>> *grid);
void squareStep(int x, int y, int stepSize, float roughness, std::vector<std::vector<Point>> *grid);

void addNoise(std::vector<std::vector<Point>>* grid, float deviation);
void scaleData(std::vector<std::vector<Point>>* grid, float factor);

void savePointData(std::vector<std::vector<Point>>* grid, std::string filename);

/***********************************
MAIN LOOP
************************************/
void main(int argc, char** argv)
{
	float noise = 0.5;
	float roughness = 1;
	float scaleFactor = 10;
	std::string filename = "data";

	if (argc > 1)
		GRID_SIZE = atoi(argv[1]) + 1;

	std::cout << "Generating points using Diamond-Square..." << std::endl;
	std::vector<std::vector<Point>> points = diamondSquare(1);

	//std::cout << "Adding noise..." << std::endl;
	//addNoise(&points, noise);

	std::cout << "Scaling points..." << std::endl;
	scaleData(&points, scaleFactor);

	std::cout << "Saving file..." << std::endl;
	savePointData(&points, filename);

	system("pause");
}

/***********************************
METHODS
************************************/
std::vector<std::vector<Point>> diamondSquare(float startDeviation)
{
	std::default_random_engine gen(time(NULL));
	std::uniform_real_distribution<float> dist(0, startDeviation);
	std::vector<std::vector<Point>> result(GRID_SIZE);


	//Initialize grid
	std::cout << "Initializing grid..." << std::endl;
	for (int i = 0; i < GRID_SIZE; i++)
	{
		result[i] = std::vector<Point>(GRID_SIZE);
		for (int j = 0; j < GRID_SIZE; j++)
		{
			result[i][j] = Point((float)i, (float)j, 0);
		}
	}

	//Set random values at the corners
	result[0][0].z = dist(gen);
	result[0][GRID_SIZE - 1].z = dist(gen);
	result[GRID_SIZE - 1][0].z = dist(gen);
	result[GRID_SIZE - 1][GRID_SIZE - 1].z = dist(gen);

	//Diamond-Square Loop
	std::cout << "Performing Diamond-Square Algorithm..." << std::endl;
	dist = std::uniform_real_distribution<float>(0, 1);
	int count = 1;
	for (int i = GRID_SIZE; i > 1; i /= 2)
	{
		for (int x = i / 2; x < GRID_SIZE; x += i)
			for(int y = i / 2; y < GRID_SIZE; y += i)
			{
				diamondStep(x, y, i/2, pow(dist(gen),count), &result);
				squareStep(x, y - i / 2, i / 2, pow(dist(gen), count), &result);
				squareStep(x - i / 2, y, i / 2, pow(dist(gen), count), &result);
				squareStep(x + i / 2, y, i / 2, pow(dist(gen), count), &result);
				squareStep(x, y + i / 2, i / 2, pow(dist(gen), count), &result);
			}
		++count;
	}

	return result;
}

void diamondStep(int x, int y, int stepSize, float r, std::vector<std::vector<Point>> *grid)
{
	float tl, tr, bl, br;
	br = (*grid)[x + stepSize][y + stepSize].z;
	bl = (*grid)[x - stepSize][y + stepSize].z;
	tr = (*grid)[x + stepSize][y - stepSize].z;
	tl = (*grid)[x - stepSize][y - stepSize].z;

	(*grid)[x][y].z = 
		((*grid)[x + stepSize][y + stepSize].z
			+ (*grid)[x - stepSize][y + stepSize].z
			+ (*grid)[x + stepSize][y - stepSize].z
			+ (*grid)[x - stepSize][y - stepSize].z) / 4 + r;
}

void squareStep(int x, int y, int stepSize, float r, std::vector<std::vector<Point>> *grid)
{
	float left = 0, right = 0, top = 0, bottom = 0;
	int count = 0;
	if (x > 0)
	{
		left = (*grid)[x - stepSize][y].z;
		count++;
	}
	if (x < GRID_SIZE - 1)
	{
		right = (*grid)[x + stepSize][y].z;
		count++;
	}
	if (y > 0)
	{
		top = (*grid)[x][y - stepSize].z;
		count++;
	}
	if (y < GRID_SIZE - 1)
	{
		bottom = (*grid)[x][y + stepSize].z;
		count++;
	}

	(*grid)[x][y].z = (left + right + top + bottom)/count + r;
}

void addNoise(std::vector<std::vector<Point>>* grid, float noise)
{
	std::default_random_engine eng;
	std::uniform_real_distribution<float> dist(-noise, noise);
	for (int i = 0; i < GRID_SIZE; i++)
	{ 
		for (int j = 0; j < GRID_SIZE; j++)
		{
			(*grid)[i][j].z += dist(eng);
			(*grid)[i][j].y += dist(eng);
			(*grid)[i][j].x += dist(eng);
		}
	}
}

void scaleData(std::vector<std::vector<Point>>* grid, float factor)
{
	for (int i = 0; i < GRID_SIZE; i++)
	{
		for (int j = 0; j < GRID_SIZE; j++)
		{
			(*grid)[i][j].z *= factor;
		}
	}
}

void savePointData(std::vector<std::vector<Point>>* grid, std::string filename)
{
	std::ofstream output("../Data/" + filename);
	
	for each (auto &row in (*grid))
	{
		for each(auto &point in row)
		{
			std::stringstream stream;
			stream << std::fixed << std::setprecision(7) << point.x << " ";
			stream << std::fixed << std::setprecision(7) << point.y << " ";
			stream << std::fixed << std::setprecision(7) << point.z << std::endl;
			output << stream.str();
		}
	}
	
	output.close();

	std::cout << "File saved in /Data/" << filename << std::endl;
}
