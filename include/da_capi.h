#ifndef DA_CAPI_H
#define DA_CAPI_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct da_ctx da_ctx;
/* ABI version. 3: added da_capi_depth_dense, da_capi_points, da_capi_free_bytes.
   4: added da_capi_load_nested (two-branch metric model).
   5: added da_capi_points_multi (fused multi-view cloud) + da_capi_gaussians.
   6: added da_capi_points_stream (sliding-window streaming cloud); removed the
      unused da_capi_depth_pose_multi. */
int         da_capi_abi_version(void);
da_ctx*     da_capi_load(const char* gguf_path, int n_threads);  /* NULL on failure */
/* Load a NESTED metric model from its two branches: the anyview (GIANT) GGUF and
   the metric (ViT-L + DPT/sky) GGUF. The returned ctx runs the nested metric
   alignment: da_capi_depth_dense / da_capi_depth_path / da_capi_pose_path all
   produce the final metric-scale depth + scaled extrinsics (is_metric=1, conf/sky
   = NULL). NULL on failure. */
da_ctx*     da_capi_load_nested(const char* anyview_gguf, const char* metric_gguf, int n_threads);
void        da_capi_free(da_ctx* ctx);                           /* safe on NULL */
/* malloc'd JSON describing model config; free via da_capi_free_string. */
char*       da_capi_info_json(da_ctx* ctx);
void        da_capi_free_string(char* s);
const char* da_capi_last_error(da_ctx* ctx);                     /* owned by ctx, "" if none */
/* Run depth on an image file. On success writes *out_h,*out_w and returns a malloc'd
   float[H*W] depth map (row-major); caller frees via da_capi_free_floats. NULL on error. */
float* da_capi_depth_path(da_ctx* ctx, const char* image_path, int* out_h, int* out_w);
void   da_capi_free_floats(float* p);
/* Run pose; fills ext[12] (3x4 row-major) and intr[9] (3x3). Returns 0 ok, -1 error. */
int da_capi_pose_path(da_ctx* ctx, const char* image_path, float out_ext[12], float out_intr[9]);
/* Single-image 3D export. Runs the native depth+pose pipeline, captures the
   processed-resolution RGB colors, and writes a glTF-2.0 binary point cloud to
   out_glb. Returns 0 ok, -1 error (see da_capi_last_error). */
int da_capi_export_glb(da_ctx* ctx, const char* image_path, const char* out_glb);
/* Single-image 3D export to a COLMAP sparse model (cameras/images/points3D) in
   directory out_dir. binary != 0 => .bin (default); 0 => .txt. Returns 0 ok, -1 error. */
int da_capi_export_colmap(da_ctx* ctx, const char* image_path, const char* out_dir, int binary);

/* Dense per-pixel output for a single image. Returns 0 ok, -1 error.
   Writes processed dims to *out_h,*out_w. Each non-NULL out_* float buffer is
   malloc'd [H*W] row-major and must be freed via da_capi_free_floats; buffers
   not produced by the model are set to NULL.
     - DualDPT model (camera-pose capable): *out_depth + *out_conf are filled,
       *out_sky = NULL, out_ext[12] (3x4 row-major) + out_intr[9] (3x3) filled.
     - mono model (DA3MONO): *out_depth + *out_sky are filled, *out_conf = NULL,
       out_ext/out_intr zeroed (mono has no camera pose).
     - nested model (da_capi_load_nested): *out_depth = final metric-scale depth,
       *out_conf = *out_sky = NULL, out_ext/out_intr = scaled extrinsics/intrinsics.
   *out_is_metric = 1 for metric/nested/mono variants (best-effort from config),
   else 0. Any of out_h/out_w/out_depth/out_conf/out_sky/out_is_metric may be NULL;
   out_ext/out_intr must point to 12/9 floats respectively. */
int da_capi_depth_dense(da_ctx* ctx, const char* image_path, int* out_h, int* out_w,
                        float** out_depth, float** out_conf, float** out_sky,
                        float out_ext[12], float out_intr[9], int* out_is_metric);

/* Single-image 3D point cloud (DualDPT/pose-capable models only; returns -1 for
   mono models with a clear last_error). Runs depth+pose+processed-RGB, back-projects
   to world space keeping pixels with conf >= conf_thresh. On success sets *out_n
   and writes a malloc'd *out_xyz[3*N float] + *out_rgb[3*N uint8]; free xyz via
   da_capi_free_floats and rgb via da_capi_free_bytes. Returns 0 ok, -1 error. */
int da_capi_points(da_ctx* ctx, const char* image_path, float conf_thresh,
                   int* out_n, float** out_xyz, unsigned char** out_rgb);
/* Free a uint8 buffer returned by da_capi_points (out_rgb). */
void da_capi_free_bytes(unsigned char* p);

/* Multi-view FUSED colored point cloud (DualDPT/pose-capable DA3 models; returns
   -1 for mono/DA2). Runs ONE cross-view depth+pose pass over n_images and back-
   projects every view into a single shared world frame, so the cloud is coherent
   across frames (no per-frame scale drift). Recovers per-pixel processed RGB and a
   perspective-correct per-point radius (~depth/focal * point_size). Keeps pixels
   with confidence >= percentile(conf, conf_pct) (conf_pct in [0,100]). Points are
   emitted grouped by source view (frames outer); if out_counts != NULL it must
   point to int[n_images] and receives the per-view point count (for progressive
   build-up rendering). On success sets *out_n and mallocs *out_xyz[3N] (world xyz,
   OpenCV frame), *out_rgb[3N] uint8, *out_radius[N]; free xyz/radius via
   da_capi_free_floats and rgb via da_capi_free_bytes. Returns 0 ok, -1 error. */
int da_capi_points_multi(da_ctx* ctx, const char** image_paths, int n_images,
                         double conf_pct, float point_size,
                         int* out_n, int* out_counts,
                         float** out_xyz, unsigned char** out_rgb, float** out_radius);

/* Sliding-window STREAMING colored point cloud (DualDPT/pose-capable DA3 models;
   returns -1 for mono/DA2). For long ordered frame sequences (time-lapse video):
   runs the fused cross-view pass on overlapping windows of chunk_size frames
   (sharing `overlap` frames) and stitches each window into ONE global frame via a
   weighted-Umeyama Sim3 solved on the overlap, so hundreds of frames fuse into a
   single coherent cloud at bounded per-pass memory. conf_pct in [0,100];
   global_budget caps total emitted points (<=0 = unlimited, subsampled per window).
   out_counts (if non-NULL) is int[n_images] receiving per-INPUT-frame point counts
   (frame-major, for progressive build-up). On success sets *out_n and mallocs
   *out_xyz[3N] (world xyz, OpenCV frame), *out_rgb[3N] uint8, *out_radius[N]; free
   xyz/radius via da_capi_free_floats and rgb via da_capi_free_bytes. 0 ok, -1 err. */
int da_capi_points_stream(da_ctx* ctx, const char** image_paths, int n_images,
                          int chunk_size, int overlap, double conf_pct,
                          float point_size, int global_budget,
                          int* out_n, int* out_counts,
                          float** out_xyz, unsigned char** out_rgb, float** out_radius);

/* Single-image 3D GAUSSIANS (DA3-GIANT / GS-head models only; returns -1 with a
   clear last_error otherwise). Returns world-frame (OpenCV) gaussians as parallel
   arrays: *out_xyz[3N] means, *out_scale[3N], *out_quat[4N] (wxyz), *out_rgb[3N]
   linear colour in [0,1] from the SH DC term, *out_opacity[N]. Sets *out_n. Free
   every returned float buffer via da_capi_free_floats. Returns 0 ok, -1 error. */
int da_capi_gaussians(da_ctx* ctx, const char* image_path, int* out_n,
                      float** out_xyz, float** out_scale, float** out_quat,
                      float** out_rgb, float** out_opacity);
#ifdef __cplusplus
}
#endif
#endif
