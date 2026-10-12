/* Standalone lookup of the firmware SM-to-GPC/TPC table.
 * Not linked into the Gold binary. Headers are the open-gpu-kernel-modules
 * tag 595.99.02 (the installed driver). The open kernel module only forwards
 * these controls; the table itself comes from the GPU firmware.
 *
 *   gcc -O2 -Wall -Wextra -o /tmp/sm_gpc_map tools/sm_gpc_map.c
 *
 * Opens /dev/nvidiactl, allocates an RM client and a subdevice, and issues
 * NV2080_CTRL_CMD_GR_GET_GPC_MASK, GET_TPC_MASK, GET_SM_TO_GPC_TPC_MAPPINGS,
 * and GET_GLOBAL_SM_ORDER. A refused control is printed with its NV_STATUS
 * and the program continues.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef uint16_t NvU16;
typedef int32_t NvS32;
typedef uint32_t NvHandle;
typedef uint32_t NvV32;
typedef void* NvP64;

#define NV_ALIGN8 __attribute__((aligned(8)))

#define NV_ESC_CHECK_VERSION_STR 210u
#define NV_ESC_RM_FREE           0x29u
#define NV_ESC_RM_CONTROL        0x2Au
#define NV_ESC_RM_ALLOC          0x2Bu
#define NV_IOCTL_MAGIC           'F'

#define NV01_ROOT_CLIENT 0x00000041u
#define NV01_DEVICE_0    0x00000080u
#define NV20_SUBDEVICE_0 0x00002080u

#define NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS 0x201u
#define NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2   0x205u
#define NV0000_CTRL_CMD_GPU_ATTACH_IDS       0x215u
#define NV0000_CTRL_GPU_MAX_ATTACHED_GPUS    32u
#define NV0000_CTRL_GPU_MAX_PROBED_GPUS      32u
#define NV0000_CTRL_GPU_INVALID_ID           0xffffffffu
#define NV_ESC_REGISTER_FD                   201u
#define NV_ESC_ATTACH_GPUS_TO_FD             212u

#define NV2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS 0x2080120fu
#define NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_MAX 240u
#define NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER       0x2080121bu
#define NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER_MAX   512u
#define NV2080_CTRL_CMD_GR_GET_GPC_MASK              0x2080122au
#define NV2080_CTRL_CMD_GR_GET_TPC_MASK              0x2080122bu

#define NV_RM_API_VERSION_CMD_STRICT 0u
#define NV_RM_API_VERSION_CMD_QUERY  '2'
#define NV_RM_API_VERSION_STRING_LENGTH 64

typedef struct {
    NvU32 cmd;
    NvU32 reply;
    char versionString[NV_RM_API_VERSION_STRING_LENGTH];
} nv_ioctl_rm_api_version_t;

typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32 hClass;
    NvP64 pAllocParms NV_ALIGN8;
    NvP64 pRightsRequested NV_ALIGN8;
    NvU32 paramsSize;
    NvU32 flags;
    NvV32 status;
} NVOS64_PARAMETERS;

typedef struct {
    NvHandle hClient;
    NvHandle hObject;
    NvV32 cmd;
    NvU32 flags;
    NvP64 params NV_ALIGN8;
    NvU32 paramsSize;
    NvV32 status;
} NVOS54_PARAMETERS;

typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectOld;
    NvV32 status;
} NVOS00_PARAMETERS;

typedef struct {
    NvU32 flags;
    NvU64 route NV_ALIGN8;
} NV0080_CTRL_GR_ROUTE_INFO;

typedef struct {
    NvU32 deviceId;
    NvHandle hClientShare;
    NvHandle hTargetClient;
    NvHandle hTargetDevice;
    NvV32 flags;
    NvU64 vaSpaceSize NV_ALIGN8;
    NvU64 vaStartInternal NV_ALIGN8;
    NvU64 vaLimitInternal NV_ALIGN8;
    NvV32 vaMode;
} NV0080_ALLOC_PARAMETERS;

typedef struct {
    NvU32 subDeviceId;
} NV2080_ALLOC_PARAMETERS;

typedef struct {
    NvU32 gpuIds[NV0000_CTRL_GPU_MAX_ATTACHED_GPUS];
} NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS;

typedef struct {
    NvU32 gpuId;
    NvU32 gpuFlags;
    NvU32 deviceInstance;
    NvU32 subDeviceInstance;
    NvU32 sliStatus;
    NvU32 boardId;
    NvU32 gpuInstance;
    NvS32 numaId;
} NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS;

typedef struct {
    NvU32 gpuIds[NV0000_CTRL_GPU_MAX_PROBED_GPUS];
    NvU32 failedId;
} NV0000_CTRL_GPU_ATTACH_IDS_PARAMS;

typedef struct {
    int ctl_fd;
} nv_ioctl_register_fd_t;

typedef struct {
    NV0080_CTRL_GR_ROUTE_INFO grRouteInfo NV_ALIGN8;
    NvU32 gpcMask;
} NV2080_CTRL_GR_GET_GPC_MASK_PARAMS;

typedef struct {
    NV0080_CTRL_GR_ROUTE_INFO grRouteInfo NV_ALIGN8;
    NvU32 gpcId;
    NvU32 tpcMask;
} NV2080_CTRL_GR_GET_TPC_MASK_PARAMS;

typedef struct {
    struct {
        NvU32 gpcId;
        NvU32 tpcId;
    } smId[NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_MAX];
    NvU32 smCount;
    NV0080_CTRL_GR_ROUTE_INFO grRouteInfo NV_ALIGN8;
} NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS;

typedef struct {
    struct {
        NvU16 gpcId;
        NvU16 localTpcId;
        NvU16 localSmId;
        NvU16 globalTpcId;
        NvU16 virtualGpcId;
        NvU16 migratableTpcId;
        NvU16 ugpuId;
        NvU16 physicalCpcId;
        NvU16 virtualTpcId;
    } globalSmId[NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER_MAX];
    NvU16 numSm;
    NvU16 numTpc;
    NV0080_CTRL_GR_ROUTE_INFO grRouteInfo NV_ALIGN8;
} NV2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS;

static int fd_ctl = -1;
static int fd_dev = -1;

static const char* nv_status_name(NvU32 s) {
    switch (s) {
    case 0x00000000u: return "NV_OK";
    case 0x00000002u: return "NV_ERR_BUFFER_TOO_SMALL";
    case 0x00000003u: return "NV_ERR_BUSY_RETRY";
    case 0x0000001au: return "NV_ERR_INSUFFICIENT_RESOURCES";
    case 0x0000001bu: return "NV_ERR_INSUFFICIENT_PERMISSIONS";
    case 0x0000001fu: return "NV_ERR_INVALID_ARGUMENT";
    case 0x00000022u: return "NV_ERR_INVALID_CLASS";
    case 0x00000023u: return "NV_ERR_INVALID_CLIENT";
    case 0x00000024u: return "NV_ERR_INVALID_COMMAND";
    case 0x00000025u: return "NV_ERR_INVALID_DATA";
    case 0x00000026u: return "NV_ERR_INVALID_DEVICE";
    case 0x00000033u: return "NV_ERR_INVALID_OBJECT_HANDLE";
    case 0x0000003au: return "NV_ERR_INVALID_PARAM_STRUCT";
    case 0x0000003bu: return "NV_ERR_INVALID_PARAMETER";
    case 0x00000040u: return "NV_ERR_INVALID_STATE";
    case 0x00000056u: return "NV_ERR_NOT_SUPPORTED";
    case 0x00000057u: return "NV_ERR_OBJECT_NOT_FOUND";
    case 0x00000059u: return "NV_ERR_OPERATING_SYSTEM";
    case 0x0000FFFFu: return "NV_ERR_GENERIC";
    default: return "NV_STATUS";
    }
}

static int rm_ioctl(unsigned nr, void* arg, size_t size) {
    unsigned long req = _IOWR(NV_IOCTL_MAGIC, nr, char);
    /* _IOWR encodes sizeof(char). Rebuild with the real size. */
    req = (req & ~((unsigned long)0x3fff << 16)) | ((unsigned long)size << 16);
    return ioctl(fd_ctl, req, arg);
}

static int rm_alloc(NvHandle root, NvHandle parent, NvHandle object, NvU32 cls,
                    void* params, NvU32 params_size, NvU32* status_out) {
    NVOS64_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hRoot = root;
    p.hObjectParent = parent;
    p.hObjectNew = object;
    p.hClass = cls;
    p.pAllocParms = params;
    p.pRightsRequested = NULL;
    p.paramsSize = params_size;
    int rc = rm_ioctl(NV_ESC_RM_ALLOC, &p, sizeof p);
    if (status_out) *status_out = p.status;
    if (rc != 0) return rc;
    return p.status == 0 ? 0 : -1;
}

static int rm_control(NvHandle client, NvHandle object, NvU32 cmd,
                      void* params, NvU32 params_size, NvU32* status_out) {
    NVOS54_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hClient = client;
    p.hObject = object;
    p.cmd = cmd;
    p.flags = 0;
    p.params = params;
    p.paramsSize = params_size;
    NvU32 st = 0xffffffffu;
    int rc = -1;
    for (int try = 0; try < 5; try++) {
        rc = rm_ioctl(NV_ESC_RM_CONTROL, &p, sizeof p);
        st = p.status;
        if (rc == 0 && st != 0x3u) break; /* NV_ERR_BUSY_RETRY */
        usleep(1000);
    }
    if (status_out) *status_out = st;
    if (rc != 0) return rc;
    return st == 0 ? 0 : -1;
}

static void report_control(const char* name, NvU32 cmd, int rc, NvU32 st) {
    printf("[ctrl] %s cmd 0x%08x ioctl_rc %d errno %d (%s) status 0x%08x (%s)\n",
           name, cmd, rc == 0 ? 0 : -1, rc == 0 ? 0 : errno, rc == 0 ? "ok" : strerror(errno),
           st, nv_status_name(st));
}

int main(void) {
    printf("sm-gpc-map: open-gpu-kernel-modules tag 595.99.02, driver lookup\n");
    printf("sizeof NVOS64 %zu NVOS54 %zu GPC_MASK %zu TPC_MASK %zu SM_MAP %zu SM_ORDER %zu\n",
           sizeof(NVOS64_PARAMETERS), sizeof(NVOS54_PARAMETERS),
           sizeof(NV2080_CTRL_GR_GET_GPC_MASK_PARAMS),
           sizeof(NV2080_CTRL_GR_GET_TPC_MASK_PARAMS),
           sizeof(NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS),
           sizeof(NV2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS));

    fd_ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (fd_ctl < 0) {
        printf("BLOCKER open /dev/nvidiactl: %s\n", strerror(errno));
        return 1;
    }

    nv_ioctl_rm_api_version_t q;
    memset(&q, 0, sizeof q);
    q.cmd = NV_RM_API_VERSION_CMD_QUERY;
    int vrc = rm_ioctl(NV_ESC_CHECK_VERSION_STR, &q, sizeof q);
    printf("[version] query ioctl_rc %d errno %d reply %u string '%s'\n",
           vrc, vrc == 0 ? 0 : errno, q.reply, q.versionString);

    nv_ioctl_rm_api_version_t ver;
    memset(&ver, 0, sizeof ver);
    ver.cmd = NV_RM_API_VERSION_CMD_STRICT;
    memcpy(ver.versionString, q.versionString[0] ? q.versionString : "595.99.02",
           NV_RM_API_VERSION_STRING_LENGTH - 1);
    vrc = rm_ioctl(NV_ESC_CHECK_VERSION_STR, &ver, sizeof ver);
    printf("[version] strict '%s' ioctl_rc %d errno %d reply %u\n",
           ver.versionString, vrc, vrc == 0 ? 0 : errno, ver.reply);
    if (vrc != 0) {
        printf("BLOCKER version check refused. Driver string from query: '%s'\n", q.versionString);
        close(fd_ctl);
        return 1;
    }

    NvHandle client = 0x5a000001u;
    NvU32 st = 0;
    int rc = rm_alloc(client, client, client, NV01_ROOT_CLIENT, NULL, 0, &st);
    printf("[alloc] root client 0x%x ioctl_rc %d errno %d status 0x%08x (%s)\n",
           client, rc == 0 ? 0 : -1, rc == 0 ? 0 : errno, st, nv_status_name(st));
    if (rc != 0) {
        printf("BLOCKER could not allocate an RM client.\n");
        close(fd_ctl);
        return 1;
    }

    NV0000_CTRL_GPU_GET_ATTACHED_IDS_PARAMS attached;
    memset(&attached, 0xff, sizeof attached);
    rc = rm_control(client, client, NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS,
                    &attached, sizeof attached, &st);
    report_control("GET_ATTACHED_IDS", NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS, rc, st);
    NvU32 gpu = NV0000_CTRL_GPU_INVALID_ID;
    if (rc == 0) {
        for (int i = 0; i < 32; i++) {
            printf("  gpuIds[%d] 0x%08x\n", i, attached.gpuIds[i]);
            if (gpu == NV0000_CTRL_GPU_INVALID_ID &&
                attached.gpuIds[i] != NV0000_CTRL_GPU_INVALID_ID &&
                attached.gpuIds[i] != 0)
                gpu = attached.gpuIds[i];
        }
    }
    if (gpu == NV0000_CTRL_GPU_INVALID_ID) {
        printf("BLOCKER no attached GPU id. Stopping before the GR controls.\n");
        goto done;
    }

    NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS info;
    memset(&info, 0, sizeof info);
    info.gpuId = gpu;
    rc = rm_control(client, client, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, &info, sizeof info, &st);
    report_control("GET_ID_INFO_V2", NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, rc, st);
    printf("  gpuId 0x%08x deviceInstance %u subDeviceInstance %u gpuInstance %u\n",
           info.gpuId, info.deviceInstance, info.subDeviceInstance, info.gpuInstance);
    if (rc != 0) {
        printf("BLOCKER GET_ID_INFO_V2 refused. GR controls not issued.\n");
        goto done;
    }

    /* This client has to attach the GPU itself. Attach does not detach
     * other clients. Do not call DETACH. */
    NV0000_CTRL_GPU_ATTACH_IDS_PARAMS att;
    memset(&att, 0xff, sizeof att);
    att.gpuIds[0] = gpu;
    att.failedId = NV0000_CTRL_GPU_INVALID_ID;
    rc = rm_control(client, client, NV0000_CTRL_CMD_GPU_ATTACH_IDS, &att, sizeof att, &st);
    report_control("ATTACH_IDS", NV0000_CTRL_CMD_GPU_ATTACH_IDS, rc, st);
    printf("  failedId 0x%08x\n", att.failedId);

    fd_dev = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
    printf("[open] /dev/nvidia0 fd %d errno %d (%s)\n", fd_dev, fd_dev < 0 ? errno : 0,
           fd_dev < 0 ? strerror(errno) : "ok");
    if (fd_dev >= 0) {
        nv_ioctl_register_fd_t reg;
        memset(&reg, 0, sizeof reg);
        reg.ctl_fd = fd_ctl;
        int rr = ioctl(fd_dev, _IOWR(NV_IOCTL_MAGIC, NV_ESC_REGISTER_FD, nv_ioctl_register_fd_t), &reg);
        printf("[ioctl] REGISTER_FD rc %d errno %d (%s)\n", rr, rr == 0 ? 0 : errno,
               rr == 0 ? "ok" : strerror(errno));
        NvU32 one = gpu;
        int ar = ioctl(fd_ctl, _IOW(NV_IOCTL_MAGIC, NV_ESC_ATTACH_GPUS_TO_FD, NvU32), &one);
        if (ar != 0) {
            /* The validator accepts an array of NvU32. Retry with _IOWR. */
            ar = ioctl(fd_ctl, _IOWR(NV_IOCTL_MAGIC, NV_ESC_ATTACH_GPUS_TO_FD, NvU32), &one);
        }
        printf("[ioctl] ATTACH_GPUS_TO_FD gpu 0x%x rc %d errno %d (%s)\n",
               one, ar, ar == 0 ? 0 : errno, ar == 0 ? "ok" : strerror(errno));
    }

    NvHandle device = 0x5a000002u;
    NV0080_ALLOC_PARAMETERS devp;
    memset(&devp, 0, sizeof devp);
    devp.deviceId = info.deviceInstance;
    devp.hClientShare = client;
    rc = rm_alloc(client, client, device, NV01_DEVICE_0, &devp, sizeof devp, &st);
    printf("[alloc] device 0x%x deviceId %u ioctl_rc %d errno %d status 0x%08x (%s)\n",
           device, devp.deviceId, rc == 0 ? 0 : -1, rc == 0 ? 0 : errno, st, nv_status_name(st));
    if (rc != 0) {
        printf("BLOCKER could not allocate the device.\n");
        goto done;
    }

    NvHandle sub = 0x5a000003u;
    NV2080_ALLOC_PARAMETERS subp;
    memset(&subp, 0, sizeof subp);
    subp.subDeviceId = info.subDeviceInstance;
    rc = rm_alloc(client, device, sub, NV20_SUBDEVICE_0, &subp, sizeof subp, &st);
    printf("[alloc] subdevice 0x%x subDeviceId %u ioctl_rc %d errno %d status 0x%08x (%s)\n",
           sub, subp.subDeviceId, rc == 0 ? 0 : -1, rc == 0 ? 0 : errno, st, nv_status_name(st));
    if (rc != 0) {
        printf("BLOCKER could not allocate the subdevice.\n");
        goto done;
    }

    NV2080_CTRL_GR_GET_GPC_MASK_PARAMS gpc;
    memset(&gpc, 0, sizeof gpc);
    rc = rm_control(client, sub, NV2080_CTRL_CMD_GR_GET_GPC_MASK, &gpc, sizeof gpc, &st);
    report_control("GET_GPC_MASK", NV2080_CTRL_CMD_GR_GET_GPC_MASK, rc, st);
    if (rc == 0) printf("  gpcMask 0x%08x\n", gpc.gpcMask);

    if (rc == 0) {
        for (unsigned g = 0; g < 32; g++) {
            if (((gpc.gpcMask >> g) & 1u) == 0) continue;
            NV2080_CTRL_GR_GET_TPC_MASK_PARAMS tpc;
            memset(&tpc, 0, sizeof tpc);
            tpc.gpcId = g;
            int tr = rm_control(client, sub, NV2080_CTRL_CMD_GR_GET_TPC_MASK, &tpc, sizeof tpc, &st);
            report_control("GET_TPC_MASK", NV2080_CTRL_CMD_GR_GET_TPC_MASK, tr, st);
            if (tr == 0) printf("  gpc %u tpcMask 0x%08x\n", g, tpc.tpcMask);
        }
    }

    static NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS map;
    memset(&map, 0, sizeof map);
    rc = rm_control(client, sub, NV2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS,
                    &map, sizeof map, &st);
    report_control("GET_SM_TO_GPC_TPC_MAPPINGS", NV2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS, rc, st);
    if (rc == 0) {
        unsigned n = map.smCount;
        if (n > NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_MAX)
            n = NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_MAX;
        printf("  smCount %u\n", map.smCount);
        printf("sm gpc tpc pass\n");
        unsigned pass_gpc[8];
        unsigned pass_seen[8];
        int pass_mixed[8];
        memset(pass_seen, 0, sizeof pass_seen);
        memset(pass_mixed, 0, sizeof pass_mixed);
        for (unsigned s = 0; s < n; s++) {
            unsigned pass = (s % 14u) / 2u;
            printf("%u %u %u %u\n", s, map.smId[s].gpcId, map.smId[s].tpcId, pass);
            if (pass < 8) {
                if (!pass_seen[pass]) {
                    pass_gpc[pass] = map.smId[s].gpcId;
                    pass_seen[pass] = 1;
                } else if (pass_gpc[pass] != map.smId[s].gpcId) {
                    pass_mixed[pass] = 1;
                }
            }
        }
        int one_gpc = 1;
        unsigned used[8];
        memset(used, 0, sizeof used);
        printf("[pass] launch pass from (smid %% 14) / 2\n");
        for (unsigned p = 0; p < 8; p++) {
            if (!pass_seen[p]) continue;
            printf("  pass %u gpc %u%s\n", p, pass_gpc[p], pass_mixed[p] ? " MIXED" : "");
            if (pass_mixed[p]) one_gpc = 0;
            if (pass_gpc[p] < 8) {
                if (used[pass_gpc[p]]) one_gpc = 0;
                used[pass_gpc[p]] = 1;
            }
        }
        printf("[question] each 12-SM pass one whole GPC, passes on different GPCs: %s\n",
               one_gpc ? "yes" : "no");
    }

    static NV2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS order;
    memset(&order, 0, sizeof order);
    rc = rm_control(client, sub, NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER,
                    &order, sizeof order, &st);
    report_control("GET_GLOBAL_SM_ORDER", NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER, rc, st);
    if (rc == 0) {
        unsigned n = order.numSm;
        if (n > NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER_MAX)
            n = NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER_MAX;
        printf("  numSm %u numTpc %u\n", order.numSm, order.numTpc);
        printf("sm gpc localTpc localSm globalTpc vGpc migTpc ugpu physCpc vTpc\n");
        for (unsigned s = 0; s < n; s++) {
            printf("%u %u %u %u %u %u %u %u %u %u\n", s,
                   order.globalSmId[s].gpcId,
                   order.globalSmId[s].localTpcId,
                   order.globalSmId[s].localSmId,
                   order.globalSmId[s].globalTpcId,
                   order.globalSmId[s].virtualGpcId,
                   order.globalSmId[s].migratableTpcId,
                   order.globalSmId[s].ugpuId,
                   order.globalSmId[s].physicalCpcId,
                   order.globalSmId[s].virtualTpcId);
        }
    }

done:
    if (fd_dev >= 0) close(fd_dev);
    if (client) {
        NVOS00_PARAMETERS fr;
        memset(&fr, 0, sizeof fr);
        fr.hRoot = client;
        fr.hObjectParent = client;
        fr.hObjectOld = client;
        rm_ioctl(NV_ESC_RM_FREE, &fr, sizeof fr);
    }
    close(fd_ctl);
    return 0;
}
