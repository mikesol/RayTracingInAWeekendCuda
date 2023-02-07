#include "json.h"
#include <iostream>
#include <time.h>
#include <sstream>
#include <string>
#include <float.h>
#include <curand_kernel.h>
#include "flatrep.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "metadata.h"
#include <fstream>
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define NUM_HITTABLES 22 * 22 + 1 + 3

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(int seed, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, 0, rand_state);
    }
}

__global__ void render_init(int seed, int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** d_list, flat_sphere* dd_list, hitable** d_world, camera** d_camera, camera* dd_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(0, vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(0, center, 0.2,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(1, center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(2, center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(2, vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(0, vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(1, vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, NUM_HITTABLES);
        float dist = 4 + 15 * RND;
        float ang = 2 * M_PI * RND;
        vec3 lookfrom(dist * sin(ang), 0.5 + (3 * RND), dist * cos(ang));
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
        memcpy(dd_camera, *d_camera, sizeof(camera));
        /////////////////////////////////////// ugggggh
        for (int ix = 0; ix < NUM_HITTABLES; ix++) {
            sphere* my_sphere = ((sphere*)d_list[ix]);
            

            if (my_sphere->tag == 0)
            {
                lambertian* lamb = (lambertian*)(my_sphere->mat_ptr);
                flat_sphere dt;
                dt.x = my_sphere->center.x();
                dt.y = my_sphere->center.y();
                dt.z = my_sphere->center.z();
                dt.radius = my_sphere->radius;
                dt.r = lamb->albedo.r();
                dt.g = lamb->albedo.g();
                dt.b = lamb->albedo.b();
                dt.lambertian_metal_dielectric = 0;
                memcpy(dd_list + ix, &dt, sizeof(flat_sphere));
                continue;
            }
            else if (my_sphere->tag == 1)
            {
                metal* mtl = (metal*)(my_sphere->mat_ptr);
                flat_sphere mt;
                mt.x = my_sphere->center.x();
                mt.y = my_sphere->center.y();
                mt.z = my_sphere->center.z();
                mt.radius = my_sphere->radius;
                mt.r = mtl->albedo.r();
                mt.g = mtl->albedo.g();
                mt.b = mtl->albedo.b();
                mt.fuzz = mtl->fuzz;
                mt.lambertian_metal_dielectric = 1;
                memcpy(dd_list + ix, &mt, sizeof(flat_sphere));
                continue;
            }
            else if (my_sphere->tag == 2)
            {
                dielectric* elex = (dielectric*)(my_sphere->mat_ptr);
                flat_sphere dl;
                dl.x = my_sphere->center.x();
                dl.y = my_sphere->center.y();
                dl.z = my_sphere->center.z();
                dl.radius = my_sphere->radius;
                dl.ref_idx = elex->ref_idx;
                dl.lambertian_metal_dielectric = 2;
                memcpy(dd_list + ix, &dl, sizeof(flat_sphere));
                continue;
            }
        }
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < NUM_HITTABLES; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

int main() {
    for (int cur_iter = 0; cur_iter < 10000; cur_iter++) {
        int nx = 1200;
        int ny = 800;
        int ns = 100;
        int tx = 8;
        int ty = 8;
        metadata out_json;
        out_json.id = cur_iter;
        out_json.width = nx;
        out_json.height = ny;
        out_json.num_samples = ns;
        out_json.cuda_tx = tx;
        out_json.cuda_ty = ty;
        std::ostringstream picStream;
        picStream << "out" << cur_iter << ".ppm";
        std::ofstream picfile(picStream.str());
        std::ostringstream jsonStream;
        jsonStream << "out" << cur_iter << ".json";
        std::ofstream jsonfile(jsonStream.str());

        std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
        std::cerr << "in " << tx << "x" << ty << " blocks.\n";

        int num_pixels = nx * ny;
        size_t fb_size = num_pixels * sizeof(vec3);

        // allocate FB
        vec3* fb;
        checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

        // allocate random state
        curandState* d_rand_state;
        checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
        curandState* d_rand_state2;
        checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

        // we need that 2nd random state to be initialized for the world creation
        rand_init << <1, 1 >> > (cur_iter, d_rand_state2);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // make our world of hitables & the camera
        int num_hitables = NUM_HITTABLES;
        hitable** d_list;
        flat_sphere* dd_list;
        flat_sphere* h_list = (flat_sphere*)malloc(num_hitables * sizeof(flat_sphere));
        checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));
        checkCudaErrors(cudaMalloc((void**)&dd_list, num_hitables * sizeof(flat_sphere)));
        hitable** d_world;
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
        camera** d_camera;
        camera* dd_camera;
        camera* h_camera = (camera*)malloc(sizeof(camera));
        checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
        checkCudaErrors(cudaMalloc((void**)&dd_camera, sizeof(camera)));
        create_world << <1, 1 >> > (d_list, dd_list, d_world, d_camera, dd_camera, nx, ny, d_rand_state2);
        
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(h_camera, dd_camera, sizeof(camera), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_list, dd_list, num_hitables * sizeof(flat_sphere), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        // write world to json
        if (true) {
            out_json.camera.aperture = h_camera->aperture;
            out_json.camera.aspect = h_camera->aspect;
            out_json.camera.focus_dist = h_camera->focus_dist;
            out_json.camera.lookat.x = h_camera->lookat.x();
            out_json.camera.lookat.y = h_camera->lookat.y();
            out_json.camera.lookat.z = h_camera->lookat.z();
            out_json.camera.lookfrom.x = h_camera->lookfrom.x();
            out_json.camera.lookfrom.y = h_camera->lookfrom.y();
            out_json.camera.lookfrom.z = h_camera->lookfrom.z();
            out_json.camera.vfov = h_camera->vfov;
            out_json.camera.vup.x = h_camera->vup.x();
            out_json.camera.vup.y = h_camera->vup.y();
            out_json.camera.vup.z = h_camera->vup.z();
            for (int ix = 0; ix < NUM_HITTABLES; ix++) {
               

                if (h_list[ix].lambertian_metal_dielectric == 0)
                {
                    lambertian_data dt;
                    dt.sphere.center.x = h_list[ix].x;
                    dt.sphere.center.y = h_list[ix].y;
                    dt.sphere.center.z = h_list[ix].z;
                    dt.sphere.radius = h_list[ix].radius;
                    dt.albedo.r = h_list[ix].r;
                    dt.albedo.g = h_list[ix].g;
                    dt.albedo.b = h_list[ix].b;
                    out_json.lambertians.push_back(dt);
                }
                else if (h_list[ix].lambertian_metal_dielectric == 1)
                {
                    metallic_data mt;
                    mt.sphere.center.x = h_list[ix].x;
                    mt.sphere.center.y = h_list[ix].y;
                    mt.sphere.center.z = h_list[ix].z;
                    mt.sphere.radius = h_list[ix].radius;
                    mt.albedo.r = h_list[ix].r;
                    mt.albedo.g = h_list[ix].g;
                    mt.albedo.b = h_list[ix].b;
                    mt.fuzz = h_list[ix].fuzz;
                    out_json.metals.push_back(mt);
                }
                 else if (h_list[ix].lambertian_metal_dielectric == 2)
                {
                    dielectric_data dl;
                    dl.sphere.center.x = h_list[ix].x;
                    dl.sphere.center.y = h_list[ix].y;
                    dl.sphere.center.z = h_list[ix].z;
                    dl.sphere.radius = h_list[ix].radius;
                    dl.ref_idx = h_list[ix].ref_idx;
                    out_json.dielectrics.push_back(dl);
                }
            }
        }

        // start rendering
        clock_t start, stop;
        start = clock();
        // Render our buffer
        dim3 blocks(nx / tx + 1, ny / ty + 1);
        dim3 threads(tx, ty);
        render_init << <blocks, threads >> > (cur_iter, nx, ny, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds." << std::endl;

        // Output FB as Image
        picfile << "P3" << std::endl << nx << " " << ny << std::endl << "255" << std::endl;
        for (int j = ny - 1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * fb[pixel_index].r());
                int ig = int(255.99 * fb[pixel_index].g());
                int ib = int(255.99 * fb[pixel_index].b());
                picfile << ir << " " << ig << " " << ib << std::endl;
            }
        }
        nlohmann::json j = out_json;
        jsonfile << j << std::endl;

        // clean up
        checkCudaErrors(cudaDeviceSynchronize());
        free_world << <1, 1 >> > (d_list, d_world, d_camera);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaFree(d_camera));
        checkCudaErrors(cudaFree(dd_camera));
        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(d_list));
        checkCudaErrors(cudaFree(dd_list));
        checkCudaErrors(cudaFree(d_rand_state));
        checkCudaErrors(cudaFree(d_rand_state2));
        checkCudaErrors(cudaFree(fb));
        free(h_list);
        free(h_camera);
        picfile.close();
        jsonfile.close();
        cudaDeviceReset();
    }
}
