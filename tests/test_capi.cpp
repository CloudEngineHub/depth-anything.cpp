#include "da_capi.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
int main(){
    const char* gguf = std::getenv("DA_TEST_GGUF");
    if (!gguf) return 77;
    if (da_capi_abi_version() != 1) return 1;
    da_ctx* c = da_capi_load(gguf, 1);
    if (!c) { std::fprintf(stderr, "load failed\n"); return 1; }
    char* j = da_capi_info_json(c);
    bool ok = j && std::strstr(j, "embed_dim");
    std::fprintf(stderr, "info json: %s -> %s\n", j ? j : "(null)", ok ? "OK" : "FAIL");
    da_capi_free_string(j);
    da_capi_free(c);
    return ok ? 0 : 1;
}
