#pragma once
#include <vector>
#include "json.h"
struct vec3_data {
	float x;
	float y;
	float z;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(vec3_data, x, y, z)
struct sphere_data {
	vec3_data center;
	float radius;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sphere_data, center, radius)
struct albedo_data {
	float r;
	float g;
	float b;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(albedo_data, r, g, b)
struct dielectric_data {
	float ref_idx;
	sphere_data sphere;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(dielectric_data, ref_idx, sphere)
struct metallic_data {
	float fuzz;
	albedo_data albedo;
	sphere_data sphere;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(metallic_data, fuzz, albedo, sphere)
struct lambertian_data {
	albedo_data albedo;
	sphere_data sphere;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(lambertian_data, albedo, sphere)
struct camera_data {
	vec3_data lookfrom;
	vec3_data lookat;
	vec3_data vup;
	float vfov;
	float aspect;
	float aperture;
	float focus_dist;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(camera_data, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist)
struct metadata {
	int id;
	int width;
	int height;
	int num_samples;
	int cuda_tx;
	int cuda_ty;
	camera_data camera;
	std::vector<lambertian_data> lambertians;
	std::vector<metallic_data> metals;
	std::vector<dielectric_data> dielectrics;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(metadata, width, height, num_samples, camera, lambertians, metals, dielectrics)