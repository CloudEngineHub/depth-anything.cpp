#ifndef DA_CAPI_H
#define DA_CAPI_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct da_ctx da_ctx;
int         da_capi_abi_version(void);
da_ctx*     da_capi_load(const char* gguf_path, int n_threads);  /* NULL on failure */
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
#ifdef __cplusplus
}
#endif
#endif
