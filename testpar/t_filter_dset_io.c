/*
 * Copyright by The HDF Group.
 * All rights reserved.
 *
 * This file is part of HDF5.  The full HDF5 copyright notice, including
 * terms governing use, modification, and redistribution, is contained in
 * the LICENSE file, which can be found at the root of the source code
 * distribution tree, or in https://www.hdfgroup.org/licenses.
 * If you do not have access to either file, you may request a copy from
 * help@hdfgroup.org.
 */

/*
 * Dataset-level filter callbacks (H5Z_class3_t dset_write / dset_read)
 * under the MPI-IO file driver.
 *
 * Each rank writes its own 8 x 8 slab of a (8 * nprocs) x 8 dataset with
 * 4 x 4 storage chunks.  The test filter compresses the slab as ONE unit,
 * stores it in the slab's first chunk with H5Dwrite_chunk() and a 4-byte
 * stub in the slab's other three chunks.  fletcher32 follows the filter in
 * the pipeline, so the unit also goes through H5Zapply_pipeline().
 *
 * The chunks each rank's callback stores are written by all ranks together
 * when the callbacks return; afterwards every rank must see every rank's
 * chunks, and the file must read back the same after reopening it, in
 * parallel and serially.
 */

#include "testpar.h"

#define FILTER_ID 542
#define DSET_NAME "dset_io"
#define FAIL_PROP "dsetio_fail"    /* the callback returns -1 on this rank */
#define DECL_PROP "dsetio_decline" /* the callback returns 0 on this rank */
#define STUB      0x42555453u      /* "STUB" */
#define SLAB      8                /* rows and columns per rank's slab */
#define CHUNK     4

static const char *FILENAME[] = {"t_filter_dset_io", NULL};

static int mpi_rank, mpi_size;

static struct {
    unsigned nwrite, nread; /* dataset-level callback invocations */
    unsigned nchunk;        /* per-chunk callback invocations */
} g_cnt;

static size_t
pass_filter(unsigned flags, size_t cd_nelmts, const unsigned *cd_values, hid_t dxpl_id, const hsize_t *scaled,
            size_t ndims, void *state, size_t nbytes, size_t *buf_size, void **buf)
{
    (void)flags;
    (void)cd_nelmts;
    (void)cd_values;
    (void)dxpl_id;
    (void)scaled;
    (void)ndims;
    (void)state;
    (void)buf_size;
    (void)buf;
    g_cnt.nchunk++;
    return nbytes;
}

/* The file selection's bounding box: this rank's slab (empty selection: 0) */
static int
slab_of(hid_t dset_id, hid_t file_space_id, hsize_t start[2], hsize_t end[2], hssize_t *npoints)
{
    hid_t sid = (H5S_ALL == file_space_id) ? H5Dget_space(dset_id) : file_space_id;
    int   ret = 0;

    if ((*npoints = H5Sget_select_npoints(sid)) < 0)
        ret = -1;
    else if (*npoints > 0 && H5Sget_select_bounds(sid, start, end) < 0)
        ret = -1;
    if (H5S_ALL == file_space_id)
        H5Sclose(sid);
    return ret;
}

static htri_t
dset_write_cb(void *state, hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
              hid_t dxpl_id, const void *buf)
{
    hsize_t  start[2], end[2];
    hssize_t npoints;
    size_t   nbytes, buf_size;
    unsigned mask = 0;
    void    *unit;
    herr_t   st = 0;

    (void)state;
    (void)mem_type_id;
    (void)mem_space_id;
    g_cnt.nwrite++;
    if (H5Pexist(dxpl_id, FAIL_PROP) > 0)
        return -1;
    if (H5Pexist(dxpl_id, DECL_PROP) > 0)
        return 0;
    if (slab_of(dset_id, file_space_id, start, end, &npoints) < 0)
        return -1;
    if (0 == npoints)
        return 1; /* nothing selected on this rank; still a participant */

    /* One unit: the whole slab (the memory buffer holds exactly the slab) */
    nbytes = buf_size = (size_t)npoints * sizeof(int);
    if (NULL == (unit = H5allocate_memory(buf_size, false)))
        return -1;
    memcpy(unit, buf, nbytes);
    if (H5Zapply_pipeline(dset_id, H5Z_PIPELINE_BEFORE, 0, dxpl_id, NULL, &mask, &nbytes, &buf_size, &unit) <
            0 ||
        H5Zapply_pipeline(dset_id, H5Z_PIPELINE_AFTER, 0, dxpl_id, NULL, &mask, &nbytes, &buf_size, &unit) <
            0)
        st = -1;

    for (hsize_t i = start[0]; i <= end[0] && st >= 0; i += CHUNK)
        for (hsize_t j = start[1]; j <= end[1] && st >= 0; j += CHUNK) {
            const hsize_t  offset[2] = {i, j};
            const unsigned stub      = STUB;
            const bool     leader    = (i == start[0] && j == start[1]);

            st = H5Dwrite_chunk(dset_id, dxpl_id, leader ? mask : 0, offset, leader ? nbytes : sizeof(stub),
                                leader ? (const void *)unit : (const void *)&stub);
        }
    H5free_memory(unit);
    return st < 0 ? -1 : 1;
}

static htri_t
dset_read_cb(void *state, hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
             hid_t dxpl_id, void *buf)
{
    hsize_t  start[2], end[2], stored = 0;
    hssize_t npoints;
    size_t   nbytes, buf_size, size;
    uint32_t mask = 0;
    void    *unit;
    herr_t   st = -1;

    (void)state;
    (void)mem_type_id;
    (void)mem_space_id;
    g_cnt.nread++;
    if (slab_of(dset_id, file_space_id, start, end, &npoints) < 0)
        return -1;
    if (0 == npoints)
        return 1;
    if (H5Dget_chunk_storage_size(dset_id, start, &stored) < 0 || 0 == stored)
        return -1;
    nbytes = buf_size = size = (size_t)stored;
    if (NULL == (unit = H5allocate_memory(buf_size, false)))
        return -1;
    if (H5Dread_chunk2(dset_id, dxpl_id, start, &mask, unit, &size) >= 0 && size == nbytes &&
        H5Zapply_pipeline(dset_id, H5Z_PIPELINE_AFTER, H5Z_FLAG_REVERSE, dxpl_id, NULL, &mask, &nbytes,
                          &buf_size, &unit) >= 0 &&
        H5Zapply_pipeline(dset_id, H5Z_PIPELINE_BEFORE, H5Z_FLAG_REVERSE, dxpl_id, NULL, &mask, &nbytes,
                          &buf_size, &unit) >= 0 &&
        nbytes == (size_t)npoints * sizeof(int)) {
        memcpy(buf, unit, nbytes);
        st = 0;
    }
    H5free_memory(unit);
    return st < 0 ? -1 : 1;
}

static const H5Z_class3_t filter_cls = {
    H5Z_CLASS3_T_VERS,       /* version */
    FILTER_ID,               /* id */
    1,                       /* encoder_present */
    1,                       /* decoder_present */
    "test_parallel_dset_io", /* name */
    NULL,                    /* can_apply */
    NULL,                    /* set_local */
    pass_filter,             /* filter */
    NULL,                    /* set_config */
    NULL,                    /* get_config */
    NULL,                    /* description */
    NULL,                    /* init */
    NULL,                    /* term */
    NULL,                    /* write_blob */
    NULL,                    /* read_blob */
    NULL,                    /* delete_blob */
    NULL,                    /* close_blob */
    dset_write_cb,           /* dset_write */
    dset_read_cb,            /* dset_read */
};

/* Value at (row, col) of the dataset */
static int
value_at(hsize_t row, hsize_t col)
{
    return (int)(row * 1000 + col);
}

/* Select rank R's slab in a copy of the dataset's dataspace */
static hid_t
slab_space(hid_t dset, int r)
{
    hsize_t start[2] = {(hsize_t)r * SLAB, 0}, count[2] = {SLAB, SLAB};
    hid_t   sid = H5Dget_space(dset);

    if (sid >= 0 && H5Sselect_hyperslab(sid, H5S_SELECT_SET, start, NULL, count, NULL) < 0) {
        H5Sclose(sid);
        sid = H5I_INVALID_HID;
    }
    return sid;
}

/* Read rank R's slab through the dataset-level path and check it */
static int
check_slab(hid_t dset, hid_t dxpl, int r)
{
    hsize_t mdims[2] = {SLAB, SLAB};
    int     rbuf[SLAB * SLAB];
    hid_t   fsid = slab_space(dset, r), msid = H5Screate_simple(2, mdims, NULL);
    int     ret = 0;

    memset(rbuf, 0, sizeof(rbuf));
    if (fsid < 0 || msid < 0 || H5Dread(dset, H5T_NATIVE_INT, msid, fsid, dxpl, rbuf) < 0)
        ret = -1;
    for (int i = 0; i < SLAB && 0 == ret; i++)
        for (int j = 0; j < SLAB; j++)
            if (rbuf[i * SLAB + j] != value_at((hsize_t)(r * SLAB + i), (hsize_t)j)) {
                ret = -1;
                break;
            }
    H5Sclose(fsid);
    H5Sclose(msid);
    return ret;
}

static int
run(void)
{
    hsize_t dims[2] = {(hsize_t)mpi_size * SLAB, SLAB}, chunk[2] = {CHUNK, CHUNK}, mdims[2] = {SLAB, SLAB};
    hid_t   fapl = H5I_INVALID_HID, file = H5I_INVALID_HID, dset = H5I_INVALID_HID;
    hid_t   dcpl = H5I_INVALID_HID, sid = H5I_INVALID_HID, fsid = H5I_INVALID_HID, msid = H5I_INVALID_HID;
    hid_t   cxpl = H5I_INVALID_HID, ixpl = H5I_INVALID_HID, bad = H5I_INVALID_HID;
    int     wbuf[SLAB * SLAB], one = 1, nerrors = 0;
    char    filename[1024];
    herr_t  ret;

    for (int i = 0; i < SLAB; i++)
        for (int j = 0; j < SLAB; j++)
            wbuf[i * SLAB + j] = value_at((hsize_t)(mpi_rank * SLAB + i), (hsize_t)j);

    if (H5Zregister(&filter_cls) < 0)
        return 1;
    if ((fapl = H5Pcreate(H5P_FILE_ACCESS)) < 0 || H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL) < 0)
        return 1;
    h5_fixname(FILENAME[0], fapl, filename, sizeof(filename));
    if ((cxpl = H5Pcreate(H5P_DATASET_XFER)) < 0 || H5Pset_dxpl_mpio(cxpl, H5FD_MPIO_COLLECTIVE) < 0)
        return 1;
    if ((ixpl = H5Pcreate(H5P_DATASET_XFER)) < 0 || H5Pset_dxpl_mpio(ixpl, H5FD_MPIO_INDEPENDENT) < 0)
        return 1;

    /* ------------------------------------------------------------------ */
    if (MAINPROCESS)
        printf("Testing parallel filter dataset I/O: one unit per rank, collective write  ");
    memset(&g_cnt, 0, sizeof(g_cnt));
    if ((file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)) < 0)
        nerrors++;
    if ((sid = H5Screate_simple(2, dims, NULL)) < 0 || (dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0 ||
        H5Pset_chunk(dcpl, 2, chunk) < 0 || H5Pset_filter(dcpl, FILTER_ID, H5Z_FLAG_MANDATORY, 0, NULL) < 0 ||
        H5Pset_fletcher32(dcpl) < 0 || H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER) < 0)
        nerrors++;
    if ((dset = H5Dcreate2(file, DSET_NAME, H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT)) < 0)
        nerrors++;
    fsid = slab_space(dset, mpi_rank);
    msid = H5Screate_simple(2, mdims, NULL);
    if (H5Dwrite(dset, H5T_NATIVE_INT, msid, fsid, cxpl, wbuf) < 0)
        nerrors++;
    /* Offered once, handled by the callback; the per-chunk path never ran */
    if (g_cnt.nwrite != 1 || g_cnt.nchunk != 0)
        nerrors++;
    /* Each rank's unit: 256 bytes + fletcher32's 4, plus three 4-byte stubs */
    if (H5Dget_storage_size(dset) != (hsize_t)mpi_size * (SLAB * SLAB * sizeof(int) + 4 + 3 * 4))
        nerrors++;
    /* Every rank sees every rank's chunks: read all slabs, collectively */
    for (int r = 0; r < mpi_size; r++)
        if (check_slab(dset, cxpl, (mpi_rank + r) % mpi_size) < 0)
            nerrors++;
    if (g_cnt.nread != (unsigned)mpi_size || g_cnt.nchunk != 0)
        nerrors++;
    /* Independent reads work too (no agreement needed) */
    if (check_slab(dset, ixpl, (mpi_rank + 1) % mpi_size) < 0)
        nerrors++;
    printf("%s", 0 == nerrors && MAINPROCESS ? "PASSED\n" : "");
    if (nerrors && MAINPROCESS)
        printf("FAILED\n");

    /* ------------------------------------------------------------------ */
    if (MAINPROCESS)
        printf("Testing parallel filter dataset I/O: overwrite, then reopen               ");
    for (int i = 0; i < SLAB * SLAB; i++)
        wbuf[i] = -wbuf[i];
    if (H5Dwrite(dset, H5T_NATIVE_INT, msid, fsid, cxpl, wbuf) < 0)
        nerrors++;
    if (H5Dclose(dset) < 0 || H5Fclose(file) < 0)
        nerrors++;
    if ((file = H5Fopen(filename, H5F_ACC_RDWR, fapl)) < 0 ||
        (dset = H5Dopen2(file, DSET_NAME, H5P_DEFAULT)) < 0)
        nerrors++;
    for (int r = 0; r < mpi_size; r++) {
        hsize_t mdims2[2] = {SLAB, SLAB};
        int     rbuf[SLAB * SLAB];
        hid_t   f2 = slab_space(dset, r), m2 = H5Screate_simple(2, mdims2, NULL);

        if (H5Dread(dset, H5T_NATIVE_INT, m2, f2, cxpl, rbuf) < 0)
            nerrors++;
        for (int i = 0; i < SLAB * SLAB; i++)
            if (rbuf[i] != -value_at((hsize_t)(r * SLAB + i / SLAB), (hsize_t)(i % SLAB))) {
                nerrors++;
                break;
            }
        H5Sclose(f2);
        H5Sclose(m2);
    }
    if (MAINPROCESS)
        printf("%s\n", nerrors ? "FAILED" : "PASSED");

    /* ------------------------------------------------------------------ */
    if (MAINPROCESS)
        printf("Testing parallel filter dataset I/O: one rank's failure fails all ranks  ");
    H5Sclose(fsid);
    fsid = slab_space(dset, mpi_rank);
    if ((bad = H5Pcopy(cxpl)) < 0)
        nerrors++;
    if (1 % mpi_size == mpi_rank &&
        H5Pinsert2(bad, FAIL_PROP, sizeof(int), &one, NULL, NULL, NULL, NULL, NULL, NULL) < 0)
        nerrors++;
    H5E_BEGIN_TRY
    {
        ret = H5Dwrite(dset, H5T_NATIVE_INT, msid, fsid, bad, wbuf);
    }
    H5E_END_TRY
    if (ret >= 0)
        nerrors++;
    H5Pclose(bad);
    if (MAINPROCESS)
        printf("%s\n", nerrors ? "FAILED" : "PASSED");

    /* ------------------------------------------------------------------ */
    if (MAINPROCESS)
        printf("Testing parallel filter dataset I/O: ranks that disagree fail             ");
    if (mpi_size > 1) {
        if ((bad = H5Pcopy(cxpl)) < 0)
            nerrors++;
        if (0 == mpi_rank &&
            H5Pinsert2(bad, DECL_PROP, sizeof(int), &one, NULL, NULL, NULL, NULL, NULL, NULL) < 0)
            nerrors++;
        H5E_BEGIN_TRY
        {
            ret = H5Dwrite(dset, H5T_NATIVE_INT, msid, fsid, bad, wbuf);
        }
        H5E_END_TRY
        if (ret >= 0)
            nerrors++;
        H5Pclose(bad);
    }
    /* The data from before the failed writes is intact */
    {
        int rbuf[SLAB * SLAB];

        if (H5Dread(dset, H5T_NATIVE_INT, msid, fsid, cxpl, rbuf) < 0 ||
            memcmp(rbuf, wbuf, sizeof(rbuf)) != 0)
            nerrors++;
    }
    if (MAINPROCESS)
        printf("%s\n", nerrors ? "FAILED" : "PASSED");

    /* ------------------------------------------------------------------ */
    if (MAINPROCESS)
        printf("Testing parallel filter dataset I/O: independent write is not offered     ");
    {
        /* Its own dataset: whether the normal path then accepts the write
         * (it does on one rank) must not disturb the dataset checked below */
        hid_t other = H5Dcreate2(file, "independent", H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT);

        memset(&g_cnt, 0, sizeof(g_cnt));
        H5E_BEGIN_TRY
        {
            (void)H5Dwrite(other, H5T_NATIVE_INT, msid, fsid, ixpl, wbuf);
        }
        H5E_END_TRY
        if (other < 0 || g_cnt.nwrite != 0)
            nerrors++;
        H5Dclose(other);
    }
    if (MAINPROCESS)
        printf("%s\n", nerrors ? "FAILED" : "PASSED");

    H5Sclose(fsid);
    H5Sclose(msid);
    H5Sclose(sid);
    H5Pclose(dcpl);
    if (H5Dclose(dset) < 0 || H5Fclose(file) < 0)
        nerrors++;

    /* ------------------------------------------------------------------ */
    if (MAINPROCESS)
        printf("Testing parallel filter dataset I/O: serial reader sees every rank's data ");
    MPI_Barrier(MPI_COMM_WORLD);
    if (MAINPROCESS) {
        hid_t sfile = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t sdset = sfile >= 0 ? H5Dopen2(sfile, DSET_NAME, H5P_DEFAULT) : H5I_INVALID_HID;
        hid_t sxpl  = H5Pcreate(H5P_DATASET_XFER);

        for (int r = 0; r < mpi_size; r++) {
            hsize_t m3[2] = {SLAB, SLAB};
            int     rbuf[SLAB * SLAB];
            hid_t   f3 = slab_space(sdset, r), mm = H5Screate_simple(2, m3, NULL);

            if (H5Dread(sdset, H5T_NATIVE_INT, mm, f3, sxpl, rbuf) < 0)
                nerrors++;
            for (int i = 0; i < SLAB * SLAB; i++)
                if (rbuf[i] != -value_at((hsize_t)(r * SLAB + i / SLAB), (hsize_t)(i % SLAB))) {
                    nerrors++;
                    break;
                }
            H5Sclose(f3);
            H5Sclose(mm);
        }
        H5Pclose(sxpl);
        H5Dclose(sdset);
        H5Fclose(sfile);
        printf("%s\n", nerrors ? "FAILED" : "PASSED");
        remove(filename);
    }

    H5Pclose(cxpl);
    H5Pclose(ixpl);
    H5Pclose(fapl);
    H5Zunregister(FILTER_ID);
    return nerrors;
}

int
main(int argc, char **argv)
{
    int nerrors, total = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    H5open();
    nerrors = run();
    MPI_Allreduce(&nerrors, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (MAINPROCESS) {
        if (total)
            printf("%d error(s) in parallel filter dataset I/O tests\n", total);
        else
            printf("All parallel filter dataset I/O tests passed.\n");
    }

    H5close();
    MPI_Finalize();
    return total ? 1 : 0;
}
