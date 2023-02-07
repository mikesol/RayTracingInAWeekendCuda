#pragma once
#pragma once
#include <vector>
#include "json.h"
struct flat_sphere {
	float x, y, z, radius;
	float r, g, b;
	float fuzz;
	float ref_idx;
	int lambertian_metal_dielectric;
};
