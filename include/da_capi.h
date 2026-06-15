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
#ifdef __cplusplus
}
#endif
#endif
