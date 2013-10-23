/*
 *   Header File for MAGMA interface
 *
 *   Taken from MAGMA Version 1.4.0
 *
 *   Reason: MAGMA does not use 'const' 
 *           in function declarations,
 *           but FLENS does
 *           -> we insert 'const' whenever 
 *           needed
 */

#ifndef PLAYGROUND_CXXMAGMA_HEADER_H
#define PLAYGROUND_CXXMAGMA_HEADER_H 1

// Custom Header to ensure const-correctness
// of MAGMA interface

#ifdef USE_CXXMAGMA

#ifdef MAGMA_ILP64
typedef int64_t INTEGER;
#else
typedef int INTEGER;
#endif

#define MagmaMaxGPUs 8

#ifdef HAVE_CUBLAS
    typedef cudaStream_t   magma_queue_t;
    typedef cudaEvent_t    magma_event_t;
    typedef int            magma_device_t;
#endif

typedef char magma_major_t;
typedef char magma_trans_t;
typedef char magma_uplo_t;
typedef char magma_diag_t;
typedef char magma_side_t;
typedef char magma_norm_t;
typedef char magma_dist_t;
typedef char magma_pack_t;
typedef char magma_vec_t;
typedef char magma_direct_t;
typedef char magma_storev_t;

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// initialization
INTEGER
magma_init( void );

INTEGER
magma_finalize( void );

void magma_version( int* major, int* minor, int* micro );


///////////////////////////////////////////////////////////////////////////////
///                  SINGLE PRECISION                                       ///
///////////////////////////////////////////////////////////////////////////////

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
INTEGER magma_get_spotrf_nb( INTEGER m );
INTEGER magma_get_sgetrf_nb( INTEGER m );
INTEGER magma_get_sgetri_nb( INTEGER m );
INTEGER magma_get_sgeqp3_nb( INTEGER m );
INTEGER magma_get_sgeqrf_nb( INTEGER m );
INTEGER magma_get_sgeqlf_nb( INTEGER m );
INTEGER magma_get_sgehrd_nb( INTEGER m );
INTEGER magma_get_ssytrd_nb( INTEGER m );
INTEGER magma_get_sgelqf_nb( INTEGER m );
INTEGER magma_get_sgebrd_nb( INTEGER m );
INTEGER magma_get_ssygst_nb( INTEGER m );
INTEGER magma_get_sgesvd_nb( INTEGER m );
INTEGER magma_get_ssygst_nb_m( INTEGER m );
INTEGER magma_get_sbulge_nb( INTEGER m, INTEGER nbthreads );
INTEGER magma_get_sbulge_nb_mgpu( INTEGER m );
INTEGER magma_sbulge_get_Vblksiz( INTEGER m, INTEGER nb, INTEGER nbthreads );
INTEGER magma_get_sbulge_gcperf();
INTEGER magma_get_smlsize_divideconquer();
/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/
void magma_smove_eig(char range, INTEGER n, float *w, INTEGER *il,
                          INTEGER *iu, float vl, float vu, INTEGER *m);
INTEGER magma_sgebrd( INTEGER m, INTEGER n, float *A,
                          INTEGER lda, float *d, float *e,
                          float *tauq,  float *taup,
                          float *work, INTEGER lwork, INTEGER *info);
INTEGER magma_sgehrd2(INTEGER n, INTEGER ilo, INTEGER ihi,
                          float *A, INTEGER lda, float *tau,
                          float *work, INTEGER lwork, INTEGER *info);
INTEGER magma_sgehrd( INTEGER n, INTEGER ilo, INTEGER ihi,
                          float *A, INTEGER lda, float *tau,
                          float *work, INTEGER lwork,
                          float *dT, INTEGER *info);
INTEGER magma_sgelqf( INTEGER m, INTEGER n,
                          float *A,    INTEGER lda,   float *tau,
                          float *work, INTEGER lwork, INTEGER *info);
INTEGER magma_sgeqlf( INTEGER m, INTEGER n,
                          float *A,    INTEGER lda,   float *tau,
                          float *work, INTEGER lwork, INTEGER *info);
INTEGER magma_sgeqrf( INTEGER m, INTEGER n, float *A,
                          INTEGER lda, float *tau, float *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_sgeqrf4(INTEGER num_gpus, INTEGER m, INTEGER n,
                          float *a,    INTEGER lda, float *tau,
                          float *work, INTEGER lwork, INTEGER *info );
INTEGER magma_sgeqrf_ooc( INTEGER m, INTEGER n, float *A,
                          INTEGER lda, float *tau, float *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_sgesv ( INTEGER n, INTEGER nrhs,
                          float *A, INTEGER lda, INTEGER *ipiv,
                          float *B, INTEGER ldb, INTEGER *info);
INTEGER magma_sgetrf( INTEGER m, INTEGER n, float *A,
                          INTEGER lda, INTEGER *ipiv,
                          INTEGER *info);
INTEGER magma_sgetrf2(INTEGER m, INTEGER n, float *a,
                          INTEGER lda, INTEGER *ipiv, INTEGER *info);

INTEGER magma_slaqps( INTEGER m, INTEGER n, INTEGER offset,
                          INTEGER nb, INTEGER *kb,
                          float *A,  INTEGER lda,
                          float *dA, INTEGER ldda,
                          INTEGER *jpvt, float *tau, float *vn1, float *vn2,
                          float *auxv,
                          float *F,  INTEGER ldf,
                          float *dF, INTEGER lddf );
void        magma_slarfg( INTEGER n, float *alpha, float *x,
                          INTEGER incx, float *tau);
INTEGER magma_slatrd( char uplo, INTEGER n, INTEGER nb, float *a,
                          INTEGER lda, float *e, float *tau,
                          float *w, INTEGER ldw,
                          float *da, INTEGER ldda,
                          float *dw, INTEGER lddw);
INTEGER magma_slatrd2(char uplo, INTEGER n, INTEGER nb,
                          float *a,  INTEGER lda,
                          float *e, float *tau,
                          float *w,  INTEGER ldw,
                          float *da, INTEGER ldda,
                          float *dw, INTEGER lddw,
                          float *dwork, INTEGER ldwork);
INTEGER magma_slahr2( INTEGER m, INTEGER n, INTEGER nb,
                          float *da, float *dv, float *a,
                          INTEGER lda, float *tau, float *t,
                          INTEGER ldt, float *y, INTEGER ldy);
INTEGER magma_slahru( INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
                          float *a, INTEGER lda,
                          float *da, float *y,
                          float *v, float *t,
                          float *dwork);
INTEGER magma_sposv ( char uplo, INTEGER n, INTEGER nrhs,
                          float *A, INTEGER lda,
                          float *B, INTEGER ldb, INTEGER *info);
INTEGER magma_spotrf( char uplo, INTEGER n, float *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_spotri( char uplo, INTEGER n, float *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_slauum( char uplo, INTEGER n, float *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_strtri( char uplo, char diag, INTEGER n, float *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_ssytrd( char uplo, INTEGER n, float *A,
                          INTEGER lda, float *d, float *e,
                          float *tau, float *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_sorgqr( INTEGER m, INTEGER n, INTEGER k,
                          float *a, INTEGER lda,
                          float *tau, float *dT,
                          INTEGER nb, INTEGER *info );
INTEGER magma_sorgqr2(INTEGER m, INTEGER n, INTEGER k,
                          float *a, INTEGER lda,
                          float *tau, INTEGER *info );
INTEGER magma_sormql( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          float *a, INTEGER lda,
                          float *tau,
                          float *c, INTEGER ldc,
                          float *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_sormqr( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          float *a, INTEGER lda, float *tau,
                          float *c, INTEGER ldc,
                          float *work, INTEGER lwork, INTEGER *info);
INTEGER magma_sormtr( char side, char uplo, char trans,
                          INTEGER m, INTEGER n,
                          float *a,    INTEGER lda,
                          float *tau,
                          float *c,    INTEGER ldc,
                          float *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_sorghr( INTEGER n, INTEGER ilo, INTEGER ihi,
                          float *a, INTEGER lda,
                          float *tau,
                          float *dT, INTEGER nb,
                          INTEGER *info);



INTEGER  magma_sgeev( char jobvl, char jobvr, INTEGER n,
                          float *a,    INTEGER lda,
                          float *wr, float *wi,
                          float *vl,   INTEGER ldvl,
                          float *vr,   INTEGER ldvr,
                          float *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_sgeqp3( INTEGER m, INTEGER n,
                          float *a, INTEGER lda,
                          INTEGER *jpvt, float *tau,
                          float *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_sgesvd( char jobu, char jobvt, INTEGER m, INTEGER n,
                          float *a,    INTEGER lda, float *s,
                          float *u,    INTEGER ldu,
                          float *vt,   INTEGER ldvt,
                          float *work, INTEGER lwork,
                          INTEGER *info );
INTEGER magma_ssyevd( char jobz, char uplo, INTEGER n,
                          float *a, INTEGER lda, float *w,
                          float *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_ssyevdx(char jobz, char range, char uplo, INTEGER n,
                          float *a, INTEGER lda,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, float *work,
                          INTEGER lwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_ssyevdx_2stage(char jobz, char range, char uplo,
                          INTEGER n,
                          float *a, INTEGER lda,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w,
                          float *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork,
                          INTEGER *info);
INTEGER magma_ssygvd( INTEGER itype, char jobz, char uplo, INTEGER n,
                          float *a, INTEGER lda,
                          float *b, INTEGER ldb,
                          float *w, float *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_ssygvdx(INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, float *a, INTEGER lda,
                          float *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, float *work,
                          INTEGER lwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_ssygvdx_2stage(INTEGER itype, char jobz, char range, char uplo, INTEGER n,
                          float *a, INTEGER lda, 
                          float *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, 
                          float *work, INTEGER lwork, 
                          INTEGER *iwork, INTEGER liwork, 
                          INTEGER *info);
INTEGER magma_sstedx( char range, INTEGER n, float vl, float vu,
                          INTEGER il, INTEGER iu, float *d, float *e,
                          float *z, INTEGER ldz,
                          float *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork,
                          float *dwork, INTEGER *info);
INTEGER magma_slaex0( INTEGER n, float *d, float *e, float *q, INTEGER ldq,
                          float *work, INTEGER *iwork, float *dwork,
                          char range, float vl, float vu,
                          INTEGER il, INTEGER iu, INTEGER *info);
INTEGER magma_slaex1( INTEGER n, float *d, float *q, INTEGER ldq,
                          INTEGER *indxq, float rho, INTEGER cutpnt,
                          float *work, INTEGER *iwork, float *dwork,
                          char range, float vl, float vu,
                          INTEGER il, INTEGER iu, INTEGER *info);
INTEGER magma_slaex3( INTEGER k, INTEGER n, INTEGER n1, float *d,
                          float *q, INTEGER ldq, float rho,
                          float *dlamda, float *q2, INTEGER *indx,
                          INTEGER *ctot, float *w, float *s, INTEGER *indxq,
                          float *dwork,
                          char range, float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *info );

INTEGER magma_ssygst( INTEGER itype, char uplo, INTEGER n,
                          float *a, INTEGER lda,
                          float *b, INTEGER ldb, INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
INTEGER magma_slahr2_m( INTEGER n, INTEGER k, INTEGER nb,
                        float *A, INTEGER lda,
                        float *tau,
                        float *T, INTEGER ldt,
                        float *Y, INTEGER ldy,
                        struct sgehrd_data *data );

INTEGER magma_slahru_m(INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
                        float *A, INTEGER lda,
                        struct sgehrd_data *data );

INTEGER magma_sgeev_m( char jobvl, char jobvr, INTEGER n,
                      float *A, INTEGER lda,
                      float *WR, float *WI,
                      float *vl, INTEGER ldvl,
                      float *vr, INTEGER ldvr,
                      float *work, INTEGER lwork,
                      INTEGER *info );


INTEGER magma_sgehrd_m( INTEGER n, INTEGER ilo, INTEGER ihi,
                        float *A, INTEGER lda,
                        float *tau,
                        float *work, INTEGER lwork,
                        float *T,
                        INTEGER *info );

INTEGER magma_sorghr_m( INTEGER n, INTEGER ilo, INTEGER ihi,
                        float *A, INTEGER lda,
                        float *tau,
                        float *T, INTEGER nb,
                        INTEGER *info );

INTEGER magma_sorgqr_m( INTEGER m, INTEGER n, INTEGER k,
                        float *A, INTEGER lda,
                        float *tau,
                        float *T, INTEGER nb,
                        INTEGER *info );

INTEGER magma_spotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            float *A, INTEGER lda,
                            INTEGER *info);
INTEGER magma_spotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            float *a, INTEGER lda, 
                            INTEGER *info);
INTEGER magma_sstedx_m( INTEGER nrgpu,
                            char range, INTEGER n, float vl, float vu,
                            INTEGER il, INTEGER iu, float *D, float *E,
                            float *Z, INTEGER ldz,
                            float *rwork, INTEGER ldrwork, INTEGER *iwork,
                            INTEGER liwork, INTEGER *info);
INTEGER magma_strsm_m ( INTEGER nrgpu,
                            char side, char uplo, char transa, char diag,
                            INTEGER m, INTEGER n, float alpha,
                            float *a, INTEGER lda, 
                            float *b, INTEGER ldb);
INTEGER magma_sormqr_m( INTEGER nrgpu, char side, char trans,
                            INTEGER m, INTEGER n, INTEGER k,
                            float *a,    INTEGER lda,
                            float *tau,
                            float *c,    INTEGER ldc,
                            float *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_sormtr_m( INTEGER nrgpu,
                            char side, char uplo, char trans,
                            INTEGER m, INTEGER n,
                            float *a,    INTEGER lda,
                            float *tau,
                            float *c,    INTEGER ldc,
                            float *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_ssygst_m( INTEGER nrgpu,
                            INTEGER itype, char uplo, INTEGER n,
                            float *a, INTEGER lda,
                            float *b, INTEGER ldb,
                            INTEGER *info);


INTEGER magma_ssyevd_m( INTEGER nrgpu, char jobz, char uplo,
                            INTEGER n,
                            float *a, INTEGER lda,
                            float *w,
                            float *work, INTEGER lwork,   
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_ssygvd_m( INTEGER nrgpu,
                            INTEGER itype, char jobz, char uplo,
                            INTEGER n,
                            float *a, INTEGER lda,
                            float *b, INTEGER ldb,
                            float *w,
                            float *work, INTEGER lwork,
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_ssyevdx_m( INTEGER nrgpu, 
                             char jobz, char range, char uplo,
                             INTEGER n,
                             float *a, INTEGER lda,
                             float vl, float vu, INTEGER il, INTEGER iu,
                             INTEGER *m, float *w,
                             float *work, INTEGER lwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_ssygvdx_m( INTEGER nrgpu,
                             INTEGER itype, char jobz, char range, char uplo,
                             INTEGER n,
                             float *a, INTEGER lda,
                             float *b, INTEGER ldb,
                             float vl, float vu, INTEGER il, INTEGER iu,
                             INTEGER *m, float *w,
                             float *work, INTEGER lwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_ssyevdx_2stage_m( INTEGER nrgpu, 
                                    char jobz, char range, char uplo,
                                    INTEGER n,
                                    float *a, INTEGER lda,
                                    float vl, float vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, float *w,
                                    float *work, INTEGER lwork,
                                    INTEGER *iwork, INTEGER liwork,
                                    INTEGER *info);
INTEGER magma_ssygvdx_2stage_m( INTEGER nrgpu, 
                                    INTEGER itype, char jobz, char range, char uplo, 
                                    INTEGER n,
                                    float *a, INTEGER lda, 
                                    float *b, INTEGER ldb,
                                    float vl, float vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, float *w, 
                                    float *work, INTEGER lwork, 
                                    INTEGER *iwork, INTEGER liwork, 
                                    INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
INTEGER magma_sgels_gpu(  char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              float *dA,    INTEGER ldda,
                              float *dB,    INTEGER lddb,
                              float *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_sgels3_gpu( char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              float *dA,    INTEGER ldda,
                              float *dB,    INTEGER lddb,
                              float *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_sgelqf_gpu( INTEGER m, INTEGER n,
                              float *dA,    INTEGER ldda,   float *tau,
                              float *work, INTEGER lwork, INTEGER *info);

INTEGER magma_sgeqr2x_gpu( INTEGER *m, INTEGER *n, float *dA,
                               INTEGER *ldda, float *dtau,
                               float *dT, float *ddA,
                               float *dwork, INTEGER *info);

INTEGER magma_sgeqr2x2_gpu( INTEGER *m, INTEGER *n, float *dA,
                                INTEGER *ldda, float *dtau,
                                float *dT, float *ddA,
                                float *dwork, INTEGER *info);

INTEGER magma_sgeqr2x3_gpu( INTEGER *m, INTEGER *n, float *dA,
                                INTEGER *ldda, float *dtau,
                                float *dT, float *ddA,
                               float *dwork, INTEGER *info);

INTEGER magma_sgeqr2x4_gpu( INTEGER *m, INTEGER *n, float *dA,
                                INTEGER *ldda, float *dtau,
                                float *dT, float *ddA,
                                float *dwork, INTEGER *info, magma_queue_t stream);

INTEGER magma_sgeqrf_gpu( INTEGER m, INTEGER n,
                              float *dA,  INTEGER ldda,
                              float *tau, float *dT,
                              INTEGER *info);
INTEGER magma_sgeqrf2_gpu(INTEGER m, INTEGER n,
                              float *dA,  INTEGER ldda,
                              float *tau, INTEGER *info);
INTEGER magma_sgeqrf2_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                               float **dlA, INTEGER ldda,
                               float *tau, INTEGER *info );
INTEGER magma_sgeqrf3_gpu(INTEGER m, INTEGER n,
                              float *dA,  INTEGER ldda,
                              float *tau, float *dT,
                              INTEGER *info);
INTEGER magma_sgeqr2_gpu( INTEGER m, INTEGER n,
                              float *dA,  INTEGER lda,
                              float *tau, float *work,
                              INTEGER *info);
INTEGER magma_sgeqrs_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              float *dA,     INTEGER ldda,
                              float *tau,   float *dT,
                              float *dB,    INTEGER lddb,
                              float *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_sgeqrs3_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              float *dA,     INTEGER ldda,
                              float *tau,   float *dT,
                              float *dB,    INTEGER lddb,
                              float *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_sgessm_gpu( char storev, INTEGER m, INTEGER n, INTEGER k, INTEGER ib,
                              INTEGER *ipiv,
                              float *dL1, INTEGER lddl1,
                              float *dL,  INTEGER lddl,
                              float *dA,  INTEGER ldda,
                              INTEGER *info);
INTEGER magma_sgesv_gpu(  INTEGER n, INTEGER nrhs,
                              float *dA, INTEGER ldda, INTEGER *ipiv,
                              float *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_sgetf2_gpu( INTEGER m, INTEGER n,
                              float *dA, INTEGER lda, INTEGER *ipiv,
                              INTEGER* info );
INTEGER magma_sgetrf_incpiv_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib,
                              float *hA, INTEGER ldha, float *dA, INTEGER ldda,
                              float *hL, INTEGER ldhl, float *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              float *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_sgetrf_gpu( INTEGER m, INTEGER n,
                              float *dA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_sgetrf_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                              float **d_lA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_sgetrf_m(INTEGER num_gpus0, INTEGER m, INTEGER n, float *a, INTEGER lda,
                           INTEGER *ipiv, INTEGER *info);
INTEGER magma_sgetrf_piv(INTEGER m, INTEGER n, INTEGER NB,
                             float *a, INTEGER lda, INTEGER *ipiv,
                             INTEGER *info);
INTEGER magma_sgetrf2_mgpu(INTEGER num_gpus,
                               INTEGER m, INTEGER n, INTEGER nb, INTEGER offset,
                               float *d_lAT[], INTEGER lddat, INTEGER *ipiv,
                               float *d_lAP[], float *a, INTEGER lda,
                               magma_queue_t streaml[][2], INTEGER *info);
INTEGER
      magma_sgetrf_nopiv_gpu( INTEGER m, INTEGER n,
                              float *dA, INTEGER ldda,
                              INTEGER *info);
INTEGER magma_sgetri_gpu( INTEGER n,
                              float *dA, INTEGER ldda, const INTEGER *ipiv,
                              float *dwork, INTEGER lwork, INTEGER *info);
INTEGER magma_sgetrs_gpu( char trans, INTEGER n, INTEGER nrhs,
                              const float *dA, INTEGER ldda, const INTEGER *ipiv,
                              float *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_slabrd_gpu( INTEGER m, INTEGER n, INTEGER nb,
                              float *a, INTEGER lda, float *da, INTEGER ldda,
                              float *d, float *e, float *tauq, float *taup,
                              float *x, INTEGER ldx, float *dx, INTEGER lddx,
                              float *y, INTEGER ldy, float *dy, INTEGER lddy);

INTEGER magma_slaqps_gpu( INTEGER m, INTEGER n, INTEGER offset,
                              INTEGER nb, INTEGER *kb,
                              float *A,  INTEGER lda,
                              INTEGER *jpvt, float *tau,
                              float *vn1, float *vn2,
                              float *auxv,
                              float *dF, INTEGER lddf);

INTEGER magma_slaqps2_gpu( INTEGER m, INTEGER n, INTEGER offset,
                                INTEGER nb, INTEGER *kb,
                                float *A,  INTEGER lda,
                                INTEGER *jpvt, float *tau,
                                float *vn1, float *vn2,
                                float *auxv,
                                float *dF, INTEGER lddf);

INTEGER magma_slaqps3_gpu( INTEGER m, INTEGER n, INTEGER offset,
                                INTEGER nb, INTEGER *kb,
                                float *A,  INTEGER lda,
                                INTEGER *jpvt, float *tau,
                                float *vn1, float *vn2,
                                float *auxv,
                                float *dF, INTEGER lddf);

INTEGER magma_slarf_gpu(  INTEGER m, INTEGER n, float *v, float *tau,
                              float *c, INTEGER ldc, float *xnorm);
INTEGER magma_slarfb_gpu( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const float *dv, INTEGER ldv,
                              const float *dt, INTEGER ldt,
                              float *dc,       INTEGER ldc,
                              float *dwork,    INTEGER ldwork );
INTEGER magma_slarfb2_gpu(INTEGER m, INTEGER n, INTEGER k,
                              const float *dV,    INTEGER ldv,
                              const float *dT,    INTEGER ldt,
                              float *dC,          INTEGER ldc,
                              float *dwork,       INTEGER ldwork );
INTEGER magma_slarfb_gpu_gemm( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const float *dv, INTEGER ldv,
                              const float *dt, INTEGER ldt,
                              float *dc,       INTEGER ldc,
                              float *dwork,    INTEGER ldwork,
                              float *dworkvt,  INTEGER ldworkvt);
INTEGER magma_slarfg_gpu( INTEGER n, float *dx0, float *dx,
                              float *dtau, float *dxnorm, float *dAkk);
INTEGER magma_sposv_gpu(  char uplo, INTEGER n, INTEGER nrhs,
                              float *dA, INTEGER ldda,
                              float *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_spotf2_gpu( magma_uplo_t uplo, INTEGER n, 
                              float *dA, INTEGER lda,
                              INTEGER *info );
INTEGER magma_spotrf_gpu( char uplo,  INTEGER n,
                              float *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_spotrf_mgpu(INTEGER ngpu, char uplo, INTEGER n,
                              float **d_lA, INTEGER ldda, INTEGER *info);
INTEGER magma_spotrf3_mgpu(INTEGER num_gpus, char uplo, INTEGER m, INTEGER n,
                               INTEGER off_i, INTEGER off_j, INTEGER nb,
                               float *d_lA[],  INTEGER ldda,
                               float *d_lP[],  INTEGER lddp,
                               float *a,      INTEGER lda,   INTEGER h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               INTEGER *info );
INTEGER magma_spotri_gpu( char uplo,  INTEGER n,
                              float *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_slauum_gpu( char uplo,  INTEGER n,
                              float *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_strtri_gpu( char uplo,  char diag, INTEGER n,
                              float *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_ssytrd_gpu( char uplo, INTEGER n,
                              float *da, INTEGER ldda,
                              float *d, float *e, float *tau,
                              float *wa,  INTEGER ldwa,
                              float *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_ssytrd2_gpu(char uplo, INTEGER n,
                              float *da, INTEGER ldda,
                              float *d, float *e, float *tau,
                              float *wa,  INTEGER ldwa,
                              float *work, INTEGER lwork,
                              float *dwork, INTEGER ldwork,
                              INTEGER *info);

float magma_slatrd_mgpu( INTEGER num_gpus, char uplo,
                          INTEGER n0, INTEGER n, INTEGER nb, INTEGER nb0,
                          float *a,  INTEGER lda,
                          float *e, float *tau,
                          float *w,   INTEGER ldw,
                          float **da, INTEGER ldda, INTEGER offset,
                          float **dw, INTEGER lddw,
                          float *dwork[MagmaMaxGPUs], INTEGER ldwork,
                          INTEGER k,
                          float  *dx[MagmaMaxGPUs], float *dy[MagmaMaxGPUs],
                          float *work,
                          magma_queue_t stream[][10],
                          float *times );

INTEGER magma_ssytrd_mgpu(INTEGER num_gpus, INTEGER k, char uplo, INTEGER n,
                              float *a, INTEGER lda,
                              float *d, float *e, float *tau,
                              float *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_ssytrd_sb2st(INTEGER threads, char uplo, 
                              INTEGER n, INTEGER nb, INTEGER Vblksiz,
                              float *A, INTEGER lda,
                              float *D, float *E,
                              float *V, INTEGER ldv,
                              float *TAU, INTEGER compT,
                              float *T, INTEGER ldt);
INTEGER magma_ssytrd_sy2sb(char uplo, INTEGER n, INTEGER NB,
                              float *a, INTEGER lda,
                              float *tau, float *work, INTEGER lwork,
                              float *dT, INTEGER threads,
                              INTEGER *info);
INTEGER magma_ssytrd_sy2sb_mgpu( char uplo, INTEGER n, INTEGER nb,
                              float *a, INTEGER lda,
                              float *tau,
                              float *work, INTEGER lwork,
                              float *dAmgpu[], INTEGER ldda,
                              float *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_ssytrd_sy2sb_mgpu_spec( char uplo, INTEGER n, INTEGER nb,
                              float *a, INTEGER lda,
                              float *tau,
                              float *work, INTEGER lwork,
                              float *dAmgpu[], INTEGER ldda,
                              float *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_spotrs_gpu( char uplo,  INTEGER n, INTEGER nrhs,
                              float *dA, INTEGER ldda,
                              float *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_sssssm_gpu( char storev, INTEGER m1, INTEGER n1,
                              INTEGER m2, INTEGER n2, INTEGER k, INTEGER ib,
                              float *dA1, INTEGER ldda1,
                              float *dA2, INTEGER ldda2,
                              float *dL1, INTEGER lddl1,
                              float *dL2, INTEGER lddl2,
                              INTEGER *IPIV, INTEGER *info);
INTEGER magma_ststrf_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib, INTEGER nb,
                              float *hU, INTEGER ldhu, float *dU, INTEGER lddu,
                              float *hA, INTEGER ldha, float *dA, INTEGER ldda,
                              float *hL, INTEGER ldhl, float *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              float *hwork, INTEGER ldhwork,
                              float *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_sorgqr_gpu( INTEGER m, INTEGER n, INTEGER k,
                              float *da, INTEGER ldda,
                              float *tau, float *dwork,
                              INTEGER nb, INTEGER *info );
INTEGER magma_sormql2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              float *da, INTEGER ldda,
                              float *tau,
                              float *dc, INTEGER lddc,
                              float *wa, INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_sormqr_gpu( char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              float *dA,    INTEGER ldda, float *tau,
                              float *dC,    INTEGER lddc,
                              float *hwork, INTEGER lwork,
                              float *dT,    INTEGER nb, INTEGER *info);
INTEGER magma_sormqr2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              float *da,   INTEGER ldda,
                              float *tau,
                              float *dc,    INTEGER lddc,
                              float *wa,    INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_sormtr_gpu( char side, char uplo, char trans,
                              INTEGER m, INTEGER n,
                              float *da,    INTEGER ldda,
                              float *tau,
                              float *dc,    INTEGER lddc,
                              float *wa,    INTEGER ldwa,
                              INTEGER *info);


INTEGER magma_sgeqp3_gpu( INTEGER m, INTEGER n,
                              float *A, INTEGER lda,
                              INTEGER *jpvt, float *tau,
                              float *work, INTEGER lwork,
                              INTEGER *info );
INTEGER magma_ssyevd_gpu( char jobz, char uplo,
                              INTEGER n,
                              float *da, INTEGER ldda,
                              float *w,
                              float *wa,  INTEGER ldwa,
                              float *work, INTEGER lwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);
INTEGER magma_ssyevdx_gpu(char jobz, char range, char uplo,
                              INTEGER n, float *da,
                              INTEGER ldda, float vl, float vu,
                              INTEGER il, INTEGER iu,
                              INTEGER *m, float *w,
                              float *wa,  INTEGER ldwa,
                              float *work, INTEGER lwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);

INTEGER magma_ssygst_gpu(INTEGER itype, char uplo, INTEGER n,
                             float *da, INTEGER ldda,
                             float *db, INTEGER lddb, INTEGER *info);


///////////////////////////////////////////////////////////////////////////////
///                  DOUBLE PRECISION                                       ///
///////////////////////////////////////////////////////////////////////////////

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
INTEGER magma_get_dpotrf_nb( INTEGER m );
INTEGER magma_get_dgetrf_nb( INTEGER m );
INTEGER magma_get_dgetri_nb( INTEGER m );
INTEGER magma_get_dgeqp3_nb( INTEGER m );
INTEGER magma_get_dgeqrf_nb( INTEGER m );
INTEGER magma_get_dgeqlf_nb( INTEGER m );
INTEGER magma_get_dgehrd_nb( INTEGER m );
INTEGER magma_get_dsytrd_nb( INTEGER m );
INTEGER magma_get_dgelqf_nb( INTEGER m );
INTEGER magma_get_dgebrd_nb( INTEGER m );
INTEGER magma_get_dsygst_nb( INTEGER m );
INTEGER magma_get_dgesvd_nb( INTEGER m );
INTEGER magma_get_dsygst_nb_m( INTEGER m );
INTEGER magma_get_dbulge_nb( INTEGER m, INTEGER nbthreads );
INTEGER magma_get_dbulge_nb_mgpu( INTEGER m );
INTEGER magma_dbulge_get_Vblksiz( INTEGER m, INTEGER nb, INTEGER nbthreads );
INTEGER magma_get_dbulge_gcperf();
INTEGER magma_get_smlsize_divideconquer();
/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/
void magma_dmove_eig(char range, INTEGER n, double *w, INTEGER *il,
                          INTEGER *iu, double vl, double vu, INTEGER *m);
INTEGER magma_dgebrd( INTEGER m, INTEGER n, double *A,
                          INTEGER lda, double *d, double *e,
                          double *tauq,  double *taup,
                          double *work, INTEGER lwork, INTEGER *info);
INTEGER magma_dgehrd2(INTEGER n, INTEGER ilo, INTEGER ihi,
                          double *A, INTEGER lda, double *tau,
                          double *work, INTEGER lwork, INTEGER *info);
INTEGER magma_dgehrd( INTEGER n, INTEGER ilo, INTEGER ihi,
                          double *A, INTEGER lda, double *tau,
                          double *work, INTEGER lwork,
                          double *dT, INTEGER *info);
INTEGER magma_dgelqf( INTEGER m, INTEGER n,
                          double *A,    INTEGER lda,   double *tau,
                          double *work, INTEGER lwork, INTEGER *info);
INTEGER magma_dgeqlf( INTEGER m, INTEGER n,
                          double *A,    INTEGER lda,   double *tau,
                          double *work, INTEGER lwork, INTEGER *info);
INTEGER magma_dgeqrf( INTEGER m, INTEGER n, double *A,
                          INTEGER lda, double *tau, double *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_dgeqrf4(INTEGER num_gpus, INTEGER m, INTEGER n,
                          double *a,    INTEGER lda, double *tau,
                          double *work, INTEGER lwork, INTEGER *info );
INTEGER magma_dgeqrf_ooc( INTEGER m, INTEGER n, double *A,
                          INTEGER lda, double *tau, double *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_dgesv ( INTEGER n, INTEGER nrhs,
                          double *A, INTEGER lda, INTEGER *ipiv,
                          double *B, INTEGER ldb, INTEGER *info);
INTEGER magma_dgetrf( INTEGER m, INTEGER n, double *A,
                          INTEGER lda, INTEGER *ipiv,
                          INTEGER *info);
INTEGER magma_dgetrf2(INTEGER m, INTEGER n, double *a,
                          INTEGER lda, INTEGER *ipiv, INTEGER *info);

INTEGER magma_dlaqps( INTEGER m, INTEGER n, INTEGER offset,
                          INTEGER nb, INTEGER *kb,
                          double *A,  INTEGER lda,
                          double *dA, INTEGER ldda,
                          INTEGER *jpvt, double *tau, double *vn1, double *vn2,
                          double *auxv,
                          double *F,  INTEGER ldf,
                          double *dF, INTEGER lddf );
void        magma_dlarfg( INTEGER n, double *alpha, double *x,
                          INTEGER incx, double *tau);
INTEGER magma_dlatrd( char uplo, INTEGER n, INTEGER nb, double *a,
                          INTEGER lda, double *e, double *tau,
                          double *w, INTEGER ldw,
                          double *da, INTEGER ldda,
                          double *dw, INTEGER lddw);
INTEGER magma_dlatrd2(char uplo, INTEGER n, INTEGER nb,
                          double *a,  INTEGER lda,
                          double *e, double *tau,
                          double *w,  INTEGER ldw,
                          double *da, INTEGER ldda,
                          double *dw, INTEGER lddw,
                          double *dwork, INTEGER ldwork);
INTEGER magma_dlahr2( INTEGER m, INTEGER n, INTEGER nb,
                          double *da, double *dv, double *a,
                          INTEGER lda, double *tau, double *t,
                          INTEGER ldt, double *y, INTEGER ldy);
INTEGER magma_dlahru( INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
                          double *a, INTEGER lda,
                          double *da, double *y,
                          double *v, double *t,
                          double *dwork);
INTEGER magma_dposv ( char uplo, INTEGER n, INTEGER nrhs,
                          double *A, INTEGER lda,
                          double *B, INTEGER ldb, INTEGER *info);
INTEGER magma_dpotrf( char uplo, INTEGER n, double *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_dpotri( char uplo, INTEGER n, double *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_dlauum( char uplo, INTEGER n, double *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_dtrtri( char uplo, char diag, INTEGER n, double *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_dsytrd( char uplo, INTEGER n, double *A,
                          INTEGER lda, double *d, double *e,
                          double *tau, double *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_dorgqr( INTEGER m, INTEGER n, INTEGER k,
                          double *a, INTEGER lda,
                          double *tau, double *dT,
                          INTEGER nb, INTEGER *info );
INTEGER magma_dorgqr2(INTEGER m, INTEGER n, INTEGER k,
                          double *a, INTEGER lda,
                          double *tau, INTEGER *info );
INTEGER magma_dormql( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          double *a, INTEGER lda,
                          double *tau,
                          double *c, INTEGER ldc,
                          double *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_dormqr( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          double *a, INTEGER lda, double *tau,
                          double *c, INTEGER ldc,
                          double *work, INTEGER lwork, INTEGER *info);
INTEGER magma_dormtr( char side, char uplo, char trans,
                          INTEGER m, INTEGER n,
                          double *a,    INTEGER lda,
                          double *tau,
                          double *c,    INTEGER ldc,
                          double *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_dorghr( INTEGER n, INTEGER ilo, INTEGER ihi,
                          double *a, INTEGER lda,
                          double *tau,
                          double *dT, INTEGER nb,
                          INTEGER *info);



INTEGER  magma_dgeev( char jobvl, char jobvr, INTEGER n,
                          double *a,    INTEGER lda,
                          double *wr, double *wi,
                          double *vl,   INTEGER ldvl,
                          double *vr,   INTEGER ldvr,
                          double *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_dgeqp3( INTEGER m, INTEGER n,
                          double *a, INTEGER lda,
                          INTEGER *jpvt, double *tau,
                          double *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_dgesvd( char jobu, char jobvt, INTEGER m, INTEGER n,
                          double *a,    INTEGER lda, double *s,
                          double *u,    INTEGER ldu,
                          double *vt,   INTEGER ldvt,
                          double *work, INTEGER lwork,
                          INTEGER *info );
INTEGER magma_dsyevd( char jobz, char uplo, INTEGER n,
                          double *a, INTEGER lda, double *w,
                          double *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_dsyevdx(char jobz, char range, char uplo, INTEGER n,
                          double *a, INTEGER lda,
                          double vl, double vu, INTEGER il, INTEGER iu,
                          INTEGER *m, double *w, double *work,
                          INTEGER lwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_dsyevdx_2stage(char jobz, char range, char uplo,
                          INTEGER n,
                          double *a, INTEGER lda,
                          double vl, double vu, INTEGER il, INTEGER iu,
                          INTEGER *m, double *w,
                          double *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork,
                          INTEGER *info);
INTEGER magma_dsygvd( INTEGER itype, char jobz, char uplo, INTEGER n,
                          double *a, INTEGER lda,
                          double *b, INTEGER ldb,
                          double *w, double *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_dsygvdx(INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, double *a, INTEGER lda,
                          double *b, INTEGER ldb,
                          double vl, double vu, INTEGER il, INTEGER iu,
                          INTEGER *m, double *w, double *work,
                          INTEGER lwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_dsygvdx_2stage(INTEGER itype, char jobz, char range, char uplo, INTEGER n,
                          double *a, INTEGER lda, 
                          double *b, INTEGER ldb,
                          double vl, double vu, INTEGER il, INTEGER iu,
                          INTEGER *m, double *w, 
                          double *work, INTEGER lwork, 
                          INTEGER *iwork, INTEGER liwork, 
                          INTEGER *info);
INTEGER magma_dstedx( char range, INTEGER n, double vl, double vu,
                          INTEGER il, INTEGER iu, double *d, double *e,
                          double *z, INTEGER ldz,
                          double *work, INTEGER lwork,
                          INTEGER *iwork, INTEGER liwork,
                          double *dwork, INTEGER *info);
INTEGER magma_dlaex0( INTEGER n, double *d, double *e, double *q, INTEGER ldq,
                          double *work, INTEGER *iwork, double *dwork,
                          char range, double vl, double vu,
                          INTEGER il, INTEGER iu, INTEGER *info);
INTEGER magma_dlaex1( INTEGER n, double *d, double *q, INTEGER ldq,
                          INTEGER *indxq, double rho, INTEGER cutpnt,
                          double *work, INTEGER *iwork, double *dwork,
                          char range, double vl, double vu,
                          INTEGER il, INTEGER iu, INTEGER *info);
INTEGER magma_dlaex3( INTEGER k, INTEGER n, INTEGER n1, double *d,
                          double *q, INTEGER ldq, double rho,
                          double *dlamda, double *q2, INTEGER *indx,
                          INTEGER *ctot, double *w, double *s, INTEGER *indxq,
                          double *dwork,
                          char range, double vl, double vu, INTEGER il, INTEGER iu,
                          INTEGER *info );

INTEGER magma_dsygst( INTEGER itype, char uplo, INTEGER n,
                          double *a, INTEGER lda,
                          double *b, INTEGER ldb, INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
INTEGER magma_dlahr2_m( INTEGER n, INTEGER k, INTEGER nb,
                        double *A, INTEGER lda,
                        double *tau,
                        double *T, INTEGER ldt,
                        double *Y, INTEGER ldy,
                        struct sgehrd_data *data );

INTEGER magma_dlahru_m( INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
                        double *A, INTEGER lda,
                        struct sgehrd_data *data );

INTEGER magma_dgeev_m( char jobvl, char jobvr, INTEGER n,
                      double *A, INTEGER lda,
                      double *WR, double *WI,
                      double *vl, INTEGER ldvl,
                      double *vr, INTEGER ldvr,
                      double *work, INTEGER lwork,
                      INTEGER *info );


INTEGER magma_dgehrd_m( INTEGER n, INTEGER ilo, INTEGER ihi,
                        double *A, INTEGER lda,
                        double *tau,
                        double *work, INTEGER lwork,
                        double *T,
                        INTEGER *info );

INTEGER magma_dorghr_m( INTEGER n, INTEGER ilo, INTEGER ihi,
                        double *A, INTEGER lda,
                        double *tau,
                        double *T, INTEGER nb,
                        INTEGER *info );

INTEGER magma_dorgqr_m( INTEGER m, INTEGER n, INTEGER k,
                        double *A, INTEGER lda,
                        double *tau,
                        double *T, INTEGER nb,
                        INTEGER *info );

INTEGER magma_dpotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            double *A, INTEGER lda,
                            INTEGER *info);
INTEGER magma_dpotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            double *a, INTEGER lda, 
                            INTEGER *info);
INTEGER magma_dstedx_m( INTEGER nrgpu,
                            char range, INTEGER n, double vl, double vu,
                            INTEGER il, INTEGER iu, double *D, double *E,
                            double *Z, INTEGER ldz,
                            double *rwork, INTEGER ldrwork, INTEGER *iwork,
                            INTEGER liwork, INTEGER *info);
INTEGER magma_dtrsm_m ( INTEGER nrgpu,
                            char side, char uplo, char transa, char diag,
                            INTEGER m, INTEGER n, double alpha,
                            double *a, INTEGER lda, 
                            double *b, INTEGER ldb);
INTEGER magma_dormqr_m( INTEGER nrgpu, char side, char trans,
                            INTEGER m, INTEGER n, INTEGER k,
                            double *a,    INTEGER lda,
                            double *tau,
                            double *c,    INTEGER ldc,
                            double *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_dormtr_m( INTEGER nrgpu,
                            char side, char uplo, char trans,
                            INTEGER m, INTEGER n,
                            double *a,    INTEGER lda,
                            double *tau,
                            double *c,    INTEGER ldc,
                            double *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_dsygst_m( INTEGER nrgpu,
                            INTEGER itype, char uplo, INTEGER n,
                            double *a, INTEGER lda,
                            double *b, INTEGER ldb,
                            INTEGER *info);


INTEGER magma_dsyevd_m( INTEGER nrgpu, char jobz, char uplo,
                            INTEGER n,
                            double *a, INTEGER lda,
                            double *w,
                            double *work, INTEGER lwork,   
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_dsygvd_m( INTEGER nrgpu,
                            INTEGER itype, char jobz, char uplo,
                            INTEGER n,
                            double *a, INTEGER lda,
                            double *b, INTEGER ldb,
                            double *w,
                            double *work, INTEGER lwork,
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_dsyevdx_m( INTEGER nrgpu, 
                             char jobz, char range, char uplo,
                             INTEGER n,
                             double *a, INTEGER lda,
                             double vl, double vu, INTEGER il, INTEGER iu,
                             INTEGER *m, double *w,
                             double *work, INTEGER lwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_dsygvdx_m( INTEGER nrgpu,
                             INTEGER itype, char jobz, char range, char uplo,
                             INTEGER n,
                             double *a, INTEGER lda,
                             double *b, INTEGER ldb,
                             double vl, double vu, INTEGER il, INTEGER iu,
                             INTEGER *m, double *w,
                             double *work, INTEGER lwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_dsyevdx_2stage_m( INTEGER nrgpu, 
                                    char jobz, char range, char uplo,
                                    INTEGER n,
                                    double *a, INTEGER lda,
                                    double vl, double vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, double *w,
                                    double *work, INTEGER lwork,
                                    INTEGER *iwork, INTEGER liwork,
                                    INTEGER *info);
INTEGER magma_dsygvdx_2stage_m( INTEGER nrgpu, 
                                    INTEGER itype, char jobz, char range, char uplo, 
                                    INTEGER n,
                                    double *a, INTEGER lda, 
                                    double *b, INTEGER ldb,
                                    double vl, double vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, double *w, 
                                    double *work, INTEGER lwork, 
                                    INTEGER *iwork, INTEGER liwork, 
                                    INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
INTEGER magma_dgels_gpu(  char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              double *dA,    INTEGER ldda,
                              double *dB,    INTEGER lddb,
                              double *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_dgels3_gpu( char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              double *dA,    INTEGER ldda,
                              double *dB,    INTEGER lddb,
                              double *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_dgelqf_gpu( INTEGER m, INTEGER n,
                              double *dA,    INTEGER ldda,   double *tau,
                              double *work, INTEGER lwork, INTEGER *info);

INTEGER magma_dgeqr2x_gpu( INTEGER *m, INTEGER *n, double *dA,
                               INTEGER *ldda, double *dtau,
                               double *dT, double *ddA,
                               double *dwork, INTEGER *info);

INTEGER magma_dgeqr2x2_gpu( INTEGER *m, INTEGER *n, double *dA,
                                INTEGER *ldda, double *dtau,
                                double *dT, double *ddA,
                                double *dwork, INTEGER *info);

INTEGER magma_dgeqr2x3_gpu( INTEGER *m, INTEGER *n, double *dA,
                                INTEGER *ldda, double *dtau,
                                double *dT, double *ddA,
                               double *dwork, INTEGER *info);

INTEGER magma_dgeqr2x4_gpu( INTEGER *m, INTEGER *n, double *dA,
                                INTEGER *ldda, double *dtau,
                                double *dT, double *ddA,
                                double *dwork, INTEGER *info, magma_queue_t stream);

INTEGER magma_dgeqrf_gpu( INTEGER m, INTEGER n,
                              double *dA,  INTEGER ldda,
                              double *tau, double *dT,
                              INTEGER *info);
INTEGER magma_dgeqrf2_gpu(INTEGER m, INTEGER n,
                              double *dA,  INTEGER ldda,
                              double *tau, INTEGER *info);
INTEGER magma_dgeqrf2_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                               double **dlA, INTEGER ldda,
                               double *tau, INTEGER *info );
INTEGER magma_dgeqrf3_gpu(INTEGER m, INTEGER n,
                              double *dA,  INTEGER ldda,
                              double *tau, double *dT,
                              INTEGER *info);
INTEGER magma_dgeqr2_gpu( INTEGER m, INTEGER n,
                              double *dA,  INTEGER lda,
                              double *tau, double *work,
                              INTEGER *info);
INTEGER magma_dgeqrs_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              double *dA,     INTEGER ldda,
                              double *tau,   double *dT,
                              double *dB,    INTEGER lddb,
                              double *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_dgeqrs3_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              double *dA,     INTEGER ldda,
                              double *tau,   double *dT,
                              double *dB,    INTEGER lddb,
                              double *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_dgessm_gpu( char storev, INTEGER m, INTEGER n, INTEGER k, INTEGER ib,
                              INTEGER *ipiv,
                              double *dL1, INTEGER lddl1,
                              double *dL,  INTEGER lddl,
                              double *dA,  INTEGER ldda,
                              INTEGER *info);
INTEGER magma_dgesv_gpu(  INTEGER n, INTEGER nrhs,
                              double *dA, INTEGER ldda, INTEGER *ipiv,
                              double *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_dgetf2_gpu( INTEGER m, INTEGER n,
                              double *dA, INTEGER lda, INTEGER *ipiv,
                              INTEGER* info );
INTEGER magma_dgetrf_incpiv_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib,
                              double *hA, INTEGER ldha, double *dA, INTEGER ldda,
                              double *hL, INTEGER ldhl, double *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              double *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_dgetrf_gpu( INTEGER m, INTEGER n,
                              double *dA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_dgetrf_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                              double **d_lA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_dgetrf_m(INTEGER num_gpus0, INTEGER m, INTEGER n, double *a, INTEGER lda,
                           INTEGER *ipiv, INTEGER *info);
INTEGER magma_dgetrf_piv(INTEGER m, INTEGER n, INTEGER NB,
                             double *a, INTEGER lda, INTEGER *ipiv,
                             INTEGER *info);
INTEGER magma_dgetrf2_mgpu(INTEGER num_gpus,
                               INTEGER m, INTEGER n, INTEGER nb, INTEGER offset,
                               double *d_lAT[], INTEGER lddat, INTEGER *ipiv,
                               double *d_lAP[], double *a, INTEGER lda,
                               magma_queue_t streaml[][2], INTEGER *info);
INTEGER
      magma_dgetrf_nopiv_gpu( INTEGER m, INTEGER n,
                              double *dA, INTEGER ldda,
                              INTEGER *info);
INTEGER magma_dgetri_gpu( INTEGER n,
                              double *dA, INTEGER ldda, const INTEGER *ipiv,
                              double *dwork, INTEGER lwork, INTEGER *info);
INTEGER magma_dgetrs_gpu( char trans, INTEGER n, INTEGER nrhs,
                              const double *dA, INTEGER ldda, const INTEGER *ipiv,
                              double *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_dlabrd_gpu( INTEGER m, INTEGER n, INTEGER nb,
                              double *a, INTEGER lda, double *da, INTEGER ldda,
                              double *d, double *e, double *tauq, double *taup,
                              double *x, INTEGER ldx, double *dx, INTEGER lddx,
                              double *y, INTEGER ldy, double *dy, INTEGER lddy);

INTEGER magma_dlaqps_gpu( INTEGER m, INTEGER n, INTEGER offset,
                              INTEGER nb, INTEGER *kb,
                              double *A,  INTEGER lda,
                              INTEGER *jpvt, double *tau,
                              double *vn1, double *vn2,
                              double *auxv,
                              double *dF, INTEGER lddf);

INTEGER magma_dlaqps2_gpu( INTEGER m, INTEGER n, INTEGER offset,
                                INTEGER nb, INTEGER *kb,
                                double *A,  INTEGER lda,
                                INTEGER *jpvt, double *tau,
                                double *vn1, double *vn2,
                                double *auxv,
                                double *dF, INTEGER lddf);

INTEGER magma_dlaqps3_gpu( INTEGER m, INTEGER n, INTEGER offset,
                                INTEGER nb, INTEGER *kb,
                                double *A,  INTEGER lda,
                                INTEGER *jpvt, double *tau,
                                double *vn1, double *vn2,
                                double *auxv,
                                double *dF, INTEGER lddf);

INTEGER magma_dlarf_gpu(  INTEGER m, INTEGER n, double *v, double *tau,
                              double *c, INTEGER ldc, double *xnorm);
INTEGER magma_dlarfb_gpu( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const double *dv, INTEGER ldv,
                              const double *dt, INTEGER ldt,
                              double *dc,       INTEGER ldc,
                              double *dwork,    INTEGER ldwork );
INTEGER magma_dlarfb2_gpu(INTEGER m, INTEGER n, INTEGER k,
                              const double *dV,    INTEGER ldv,
                              const double *dT,    INTEGER ldt,
                              double *dC,          INTEGER ldc,
                              double *dwork,       INTEGER ldwork );
INTEGER magma_dlarfb_gpu_gemm( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const double *dv, INTEGER ldv,
                              const double *dt, INTEGER ldt,
                              double *dc,       INTEGER ldc,
                              double *dwork,    INTEGER ldwork,
                              double *dworkvt,  INTEGER ldworkvt);
INTEGER magma_dlarfg_gpu( INTEGER n, double *dx0, double *dx,
                              double *dtau, double *dxnorm, double *dAkk);
INTEGER magma_dposv_gpu(  char uplo, INTEGER n, INTEGER nrhs,
                              double *dA, INTEGER ldda,
                              double *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_dpotf2_gpu( magma_uplo_t uplo, INTEGER n, 
                              double *dA, INTEGER lda,
                              INTEGER *info );
INTEGER magma_dpotrf_gpu( char uplo,  INTEGER n,
                              double *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_dpotrf_mgpu(INTEGER ngpu, char uplo, INTEGER n,
                              double **d_lA, INTEGER ldda, INTEGER *info);
INTEGER magma_dpotrf3_mgpu(INTEGER num_gpus, char uplo, INTEGER m, INTEGER n,
                               INTEGER off_i, INTEGER off_j, INTEGER nb,
                               double *d_lA[],  INTEGER ldda,
                               double *d_lP[],  INTEGER lddp,
                               double *a,      INTEGER lda,   INTEGER h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               INTEGER *info );
INTEGER magma_dpotri_gpu( char uplo,  INTEGER n,
                              double *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_dlauum_gpu( char uplo,  INTEGER n,
                              double *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_dtrtri_gpu( char uplo,  char diag, INTEGER n,
                              double *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_dsytrd_gpu( char uplo, INTEGER n,
                              double *da, INTEGER ldda,
                              double *d, double *e, double *tau,
                              double *wa,  INTEGER ldwa,
                              double *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_dsytrd2_gpu(char uplo, INTEGER n,
                              double *da, INTEGER ldda,
                              double *d, double *e, double *tau,
                              double *wa,  INTEGER ldwa,
                              double *work, INTEGER lwork,
                              double *dwork, INTEGER ldwork,
                              INTEGER *info);

double magma_dlatrd_mgpu( INTEGER num_gpus, char uplo,
                          INTEGER n0, INTEGER n, INTEGER nb, INTEGER nb0,
                          double *a,  INTEGER lda,
                          double *e, double *tau,
                          double *w,   INTEGER ldw,
                          double **da, INTEGER ldda, INTEGER offset,
                          double **dw, INTEGER lddw,
                          double *dwork[MagmaMaxGPUs], INTEGER ldwork,
                          INTEGER k,
                          double  *dx[MagmaMaxGPUs], double *dy[MagmaMaxGPUs],
                          double *work,
                          magma_queue_t stream[][10],
                          double *times );

INTEGER magma_dsytrd_mgpu(INTEGER num_gpus, INTEGER k, char uplo, INTEGER n,
                              double *a, INTEGER lda,
                              double *d, double *e, double *tau,
                              double *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_dsytrd_sb2st(INTEGER threads, char uplo, 
                              INTEGER n, INTEGER nb, INTEGER Vblksiz,
                              double *A, INTEGER lda,
                              double *D, double *E,
                              double *V, INTEGER ldv,
                              double *TAU, INTEGER compT,
                              double *T, INTEGER ldt);
INTEGER magma_dsytrd_sy2sb(char uplo, INTEGER n, INTEGER NB,
                              double *a, INTEGER lda,
                              double *tau, double *work, INTEGER lwork,
                              double *dT, INTEGER threads,
                              INTEGER *info);
INTEGER magma_dsytrd_sy2sb_mgpu( char uplo, INTEGER n, INTEGER nb,
                              double *a, INTEGER lda,
                              double *tau,
                              double *work, INTEGER lwork,
                              double *dAmgpu[], INTEGER ldda,
                              double *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_dsytrd_sy2sb_mgpu_spec( char uplo, INTEGER n, INTEGER nb,
                              double *a, INTEGER lda,
                              double *tau,
                              double *work, INTEGER lwork,
                              double *dAmgpu[], INTEGER ldda,
                              double *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_dpotrs_gpu( char uplo,  INTEGER n, INTEGER nrhs,
                              double *dA, INTEGER ldda,
                              double *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_dssssm_gpu( char storev, INTEGER m1, INTEGER n1,
                              INTEGER m2, INTEGER n2, INTEGER k, INTEGER ib,
                              double *dA1, INTEGER ldda1,
                              double *dA2, INTEGER ldda2,
                              double *dL1, INTEGER lddl1,
                              double *dL2, INTEGER lddl2,
                              INTEGER *IPIV, INTEGER *info);
INTEGER magma_dtstrf_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib, INTEGER nb,
                              double *hU, INTEGER ldhu, double *dU, INTEGER lddu,
                              double *hA, INTEGER ldha, double *dA, INTEGER ldda,
                              double *hL, INTEGER ldhl, double *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              double *hwork, INTEGER ldhwork,
                              double *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_dorgqr_gpu( INTEGER m, INTEGER n, INTEGER k,
                              double *da, INTEGER ldda,
                              double *tau, double *dwork,
                              INTEGER nb, INTEGER *info );
INTEGER magma_dormql2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              double *da, INTEGER ldda,
                              double *tau,
                              double *dc, INTEGER lddc,
                              double *wa, INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_dormqr_gpu( char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              double *dA,    INTEGER ldda, double *tau,
                              double *dC,    INTEGER lddc,
                              double *hwork, INTEGER lwork,
                              double *dT,    INTEGER nb, INTEGER *info);
INTEGER magma_dormqr2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              double *da,   INTEGER ldda,
                              double *tau,
                              double *dc,    INTEGER lddc,
                              double *wa,    INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_dormtr_gpu( char side, char uplo, char trans,
                              INTEGER m, INTEGER n,
                              double *da,    INTEGER ldda,
                              double *tau,
                              double *dc,    INTEGER lddc,
                              double *wa,    INTEGER ldwa,
                              INTEGER *info);


INTEGER magma_dgeqp3_gpu( INTEGER m, INTEGER n,
                              double *A, INTEGER lda,
                              INTEGER *jpvt, double *tau,
                              double *work, INTEGER lwork,
                              INTEGER *info );
INTEGER magma_dsyevd_gpu( char jobz, char uplo,
                              INTEGER n,
                              double *da, INTEGER ldda,
                              double *w,
                              double *wa,  INTEGER ldwa,
                              double *work, INTEGER lwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);
INTEGER magma_dsyevdx_gpu(char jobz, char range, char uplo,
                              INTEGER n, double *da,
                              INTEGER ldda, double vl, double vu,
                              INTEGER il, INTEGER iu,
                              INTEGER *m, double *w,
                              double *wa,  INTEGER ldwa,
                              double *work, INTEGER lwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);

INTEGER magma_dsygst_gpu(INTEGER itype, char uplo, INTEGER n,
                             double *da, INTEGER ldda,
                             double *db, INTEGER lddb, INTEGER *info);




/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
INTEGER magma_get_cpotrf_nb( INTEGER m );
INTEGER magma_get_cgetrf_nb( INTEGER m );
INTEGER magma_get_cgetri_nb( INTEGER m );
INTEGER magma_get_cgeqp3_nb( INTEGER m );
INTEGER magma_get_cgeqrf_nb( INTEGER m );
INTEGER magma_get_cgeqlf_nb( INTEGER m );
INTEGER magma_get_cgehrd_nb( INTEGER m );
INTEGER magma_get_chetrd_nb( INTEGER m );
INTEGER magma_get_cgelqf_nb( INTEGER m );
INTEGER magma_get_cgebrd_nb( INTEGER m );
INTEGER magma_get_chegst_nb( INTEGER m );
INTEGER magma_get_cgesvd_nb( INTEGER m );
INTEGER magma_get_chegst_nb_m( INTEGER m );
INTEGER magma_get_cbulge_nb( INTEGER m, INTEGER nbthreads );
INTEGER magma_get_cbulge_nb_mgpu( INTEGER m );
INTEGER magma_cbulge_get_Vblksiz( INTEGER m, INTEGER nb, INTEGER nbthreads );
INTEGER magma_get_cbulge_gcperf();
INTEGER magma_get_smlsize_divideconquer();
/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/
void magma_smove_eig(char range, INTEGER n, float *w, INTEGER *il,
                          INTEGER *iu, float vl, float vu, INTEGER *m);
INTEGER magma_cgebrd( INTEGER m, INTEGER n, std::complex<float> *A,
                          INTEGER lda, float *d, float *e,
                          std::complex<float> *tauq,  std::complex<float> *taup,
                          std::complex<float> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_cgehrd2(INTEGER n, INTEGER ilo, INTEGER ihi,
                          std::complex<float> *A, INTEGER lda, std::complex<float> *tau,
                          std::complex<float> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_cgehrd( INTEGER n, INTEGER ilo, INTEGER ihi,
                          std::complex<float> *A, INTEGER lda, std::complex<float> *tau,
                          std::complex<float> *work, INTEGER lwork,
                          std::complex<float> *dT, INTEGER *info);
INTEGER magma_cgelqf( INTEGER m, INTEGER n,
                          std::complex<float> *A,    INTEGER lda,   std::complex<float> *tau,
                          std::complex<float> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_cgeqlf( INTEGER m, INTEGER n,
                          std::complex<float> *A,    INTEGER lda,   std::complex<float> *tau,
                          std::complex<float> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_cgeqrf( INTEGER m, INTEGER n, std::complex<float> *A,
                          INTEGER lda, std::complex<float> *tau, std::complex<float> *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_cgeqrf4(INTEGER num_gpus, INTEGER m, INTEGER n,
                          std::complex<float> *a,    INTEGER lda, std::complex<float> *tau,
                          std::complex<float> *work, INTEGER lwork, INTEGER *info );
INTEGER magma_cgeqrf_ooc( INTEGER m, INTEGER n, std::complex<float> *A,
                          INTEGER lda, std::complex<float> *tau, std::complex<float> *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_cgesv ( INTEGER n, INTEGER nrhs,
                          std::complex<float> *A, INTEGER lda, INTEGER *ipiv,
                          std::complex<float> *B, INTEGER ldb, INTEGER *info);
INTEGER magma_cgetrf( INTEGER m, INTEGER n, std::complex<float> *A,
                          INTEGER lda, INTEGER *ipiv,
                          INTEGER *info);
INTEGER magma_cgetrf2(INTEGER m, INTEGER n, std::complex<float> *a,
                          INTEGER lda, INTEGER *ipiv, INTEGER *info);

INTEGER magma_claqps( INTEGER m, INTEGER n, INTEGER offset,
                          INTEGER nb, INTEGER *kb,
                          std::complex<float> *A,  INTEGER lda,
                          std::complex<float> *dA, INTEGER ldda,
                          INTEGER *jpvt, std::complex<float> *tau, float *vn1, float *vn2,
                          std::complex<float> *auxv,
                          std::complex<float> *F,  INTEGER ldf,
                          std::complex<float> *dF, INTEGER lddf );
void        magma_clarfg( INTEGER n, std::complex<float> *alpha, std::complex<float> *x,
                          INTEGER incx, std::complex<float> *tau);
INTEGER magma_clatrd( char uplo, INTEGER n, INTEGER nb, std::complex<float> *a,
                          INTEGER lda, float *e, std::complex<float> *tau,
                          std::complex<float> *w, INTEGER ldw,
                          std::complex<float> *da, INTEGER ldda,
                          std::complex<float> *dw, INTEGER lddw);
INTEGER magma_clatrd2(char uplo, INTEGER n, INTEGER nb,
                          std::complex<float> *a,  INTEGER lda,
                          float *e, std::complex<float> *tau,
                          std::complex<float> *w,  INTEGER ldw,
                          std::complex<float> *da, INTEGER ldda,
                          std::complex<float> *dw, INTEGER lddw,
                          std::complex<float> *dwork, INTEGER ldwork);
INTEGER magma_clahr2( INTEGER m, INTEGER n, INTEGER nb,
                          std::complex<float> *da, std::complex<float> *dv, std::complex<float> *a,
                          INTEGER lda, std::complex<float> *tau, std::complex<float> *t,
                          INTEGER ldt, std::complex<float> *y, INTEGER ldy);
INTEGER magma_clahru( INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *da, std::complex<float> *y,
                          std::complex<float> *v, std::complex<float> *t,
                          std::complex<float> *dwork);
INTEGER magma_cposv ( char uplo, INTEGER n, INTEGER nrhs,
                          std::complex<float> *A, INTEGER lda,
                          std::complex<float> *B, INTEGER ldb, INTEGER *info);
INTEGER magma_cpotrf( char uplo, INTEGER n, std::complex<float> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_cpotri( char uplo, INTEGER n, std::complex<float> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_clauum( char uplo, INTEGER n, std::complex<float> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_ctrtri( char uplo, char diag, INTEGER n, std::complex<float> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_chetrd( char uplo, INTEGER n, std::complex<float> *A,
                          INTEGER lda, float *d, float *e,
                          std::complex<float> *tau, std::complex<float> *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_cungqr( INTEGER m, INTEGER n, INTEGER k,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *tau, std::complex<float> *dT,
                          INTEGER nb, INTEGER *info );
INTEGER magma_cungqr2(INTEGER m, INTEGER n, INTEGER k,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *tau, INTEGER *info );
INTEGER magma_cunmql( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *tau,
                          std::complex<float> *c, INTEGER ldc,
                          std::complex<float> *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_cunmqr( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          std::complex<float> *a, INTEGER lda, std::complex<float> *tau,
                          std::complex<float> *c, INTEGER ldc,
                          std::complex<float> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_cunmtr( char side, char uplo, char trans,
                          INTEGER m, INTEGER n,
                          std::complex<float> *a,    INTEGER lda,
                          std::complex<float> *tau,
                          std::complex<float> *c,    INTEGER ldc,
                          std::complex<float> *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_cunghr( INTEGER n, INTEGER ilo, INTEGER ihi,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *tau,
                          std::complex<float> *dT, INTEGER nb,
                          INTEGER *info);


INTEGER  magma_cgeev( char jobvl, char jobvr, INTEGER n,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *w,
                          std::complex<float> *vl, INTEGER ldvl,
                          std::complex<float> *vr, INTEGER ldvr,
                          std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER *info);
INTEGER magma_cgeqp3( INTEGER m, INTEGER n,
                          std::complex<float> *a, INTEGER lda,
                          INTEGER *jpvt, std::complex<float> *tau,
                          std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER *info);
INTEGER magma_cgesvd( char jobu, char jobvt, INTEGER m, INTEGER n,
                          std::complex<float> *a,    INTEGER lda, float *s,
                          std::complex<float> *u,    INTEGER ldu,
                          std::complex<float> *vt,   INTEGER ldvt,
                          std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER *info );
INTEGER magma_cheevd( char jobz, char uplo, INTEGER n,
                          std::complex<float> *a, INTEGER lda, float *w,
                          std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_cheevdx(char jobz, char range, char uplo, INTEGER n,
                          std::complex<float> *a, INTEGER lda,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, std::complex<float> *work,
                          INTEGER lwork, float *rwork, INTEGER lrwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_cheevdx_2stage(char jobz, char range, char uplo,
                          INTEGER n,
                          std::complex<float> *a, INTEGER lda,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w,
                          std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork,
                          INTEGER *iwork, INTEGER liwork,
                          INTEGER *info);
INTEGER magma_cheevx( char jobz, char range, char uplo, INTEGER n,
                          std::complex<float> *a, INTEGER lda, float vl, float vu,
                          INTEGER il, INTEGER iu, float abstol, INTEGER *m,
                          float *w, std::complex<float> *z, INTEGER ldz,
                          std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER *iwork,
                          INTEGER *ifail, INTEGER *info);
INTEGER magma_cheevr( char jobz, char range, char uplo, INTEGER n,
                          std::complex<float> *a, INTEGER lda, float vl, float vu,
                          INTEGER il, INTEGER iu, float abstol, INTEGER *m,
                          float *w, std::complex<float> *z, INTEGER ldz,
                          INTEGER *isuppz,
                          std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_chegvd( INTEGER itype, char jobz, char uplo, INTEGER n,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *b, INTEGER ldb,
                          float *w, std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_chegvdx(INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, std::complex<float> *a, INTEGER lda,
                          std::complex<float> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, std::complex<float> *work,
                          INTEGER lwork, float *rwork,
                          INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_chegvdx_2stage(INTEGER itype, char jobz, char range, char uplo, INTEGER n,
                          std::complex<float> *a, INTEGER lda, 
                          std::complex<float> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, 
                          std::complex<float> *work, INTEGER lwork, 
                          float *rwork, INTEGER lrwork, 
                          INTEGER *iwork, INTEGER liwork, 
                          INTEGER *info);
INTEGER magma_chegvx( INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, std::complex<float> *a, INTEGER lda,
                          std::complex<float> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          float abstol, INTEGER *m, float *w,
                          std::complex<float> *z, INTEGER ldz,
                          std::complex<float> *work, INTEGER lwork, float *rwork,
                          INTEGER *iwork, INTEGER *ifail, INTEGER *info);
INTEGER magma_chegvr( INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, std::complex<float> *a, INTEGER lda,
                          std::complex<float> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          float abstol, INTEGER *m, float *w,
                          std::complex<float> *z, INTEGER ldz,
                          INTEGER *isuppz, std::complex<float> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_cstedx( char range, INTEGER n, float vl, float vu,
                          INTEGER il, INTEGER iu, float *D, float *E,
                          std::complex<float> *Z, INTEGER ldz,
                          float *rwork, INTEGER ldrwork, INTEGER *iwork,
                          INTEGER liwork, float *dwork, INTEGER *info);



INTEGER magma_chegst( INTEGER itype, char uplo, INTEGER n,
                          std::complex<float> *a, INTEGER lda,
                          std::complex<float> *b, INTEGER ldb, INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
INTEGER magma_clahr2_m(
    INTEGER n, INTEGER k, INTEGER nb,
    std::complex<float> *A, INTEGER lda,
    std::complex<float> *tau,
    std::complex<float> *T, INTEGER ldt,
    std::complex<float> *Y, INTEGER ldy,
    struct cgehrd_data *data );

INTEGER magma_clahru_m(
    INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
    std::complex<float> *A, INTEGER lda,
    struct cgehrd_data *data );

INTEGER magma_cgeev_m(
    char jobvl, char jobvr, INTEGER n,
    std::complex<float> *A, INTEGER lda,
    std::complex<float> *W,
    std::complex<float> *vl, INTEGER ldvl,
    std::complex<float> *vr, INTEGER ldvr,
    std::complex<float> *work, INTEGER lwork,
    float *rwork,
    INTEGER *info );


INTEGER magma_cgehrd_m(
    INTEGER n, INTEGER ilo, INTEGER ihi,
    std::complex<float> *A, INTEGER lda,
    std::complex<float> *tau,
    std::complex<float> *work, INTEGER lwork,
    std::complex<float> *T,
    INTEGER *info );

INTEGER magma_cunghr_m(
    INTEGER n, INTEGER ilo, INTEGER ihi,
    std::complex<float> *A, INTEGER lda,
    std::complex<float> *tau,
    std::complex<float> *T, INTEGER nb,
    INTEGER *info );

INTEGER magma_cungqr_m(
    INTEGER m, INTEGER n, INTEGER k,
    std::complex<float> *A, INTEGER lda,
    std::complex<float> *tau,
    std::complex<float> *T, INTEGER nb,
    INTEGER *info );

INTEGER magma_cpotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            std::complex<float> *A, INTEGER lda,
                            INTEGER *info);
INTEGER magma_cpotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            std::complex<float> *a, INTEGER lda, 
                            INTEGER *info);
INTEGER magma_cstedx_m( INTEGER nrgpu,
                            char range, INTEGER n, float vl, float vu,
                            INTEGER il, INTEGER iu, float *D, float *E,
                            std::complex<float> *Z, INTEGER ldz,
                            float *rwork, INTEGER ldrwork, INTEGER *iwork,
                            INTEGER liwork, INTEGER *info);
INTEGER magma_ctrsm_m ( INTEGER nrgpu,
                            char side, char uplo, char transa, char diag,
                            INTEGER m, INTEGER n, std::complex<float> alpha,
                            std::complex<float> *a, INTEGER lda, 
                            std::complex<float> *b, INTEGER ldb);
INTEGER magma_cunmqr_m( INTEGER nrgpu, char side, char trans,
                            INTEGER m, INTEGER n, INTEGER k,
                            std::complex<float> *a,    INTEGER lda,
                            std::complex<float> *tau,
                            std::complex<float> *c,    INTEGER ldc,
                            std::complex<float> *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_cunmtr_m( INTEGER nrgpu,
                            char side, char uplo, char trans,
                            INTEGER m, INTEGER n,
                            std::complex<float> *a,    INTEGER lda,
                            std::complex<float> *tau,
                            std::complex<float> *c,    INTEGER ldc,
                            std::complex<float> *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_chegst_m( INTEGER nrgpu,
                            INTEGER itype, char uplo, INTEGER n,
                            std::complex<float> *a, INTEGER lda,
                            std::complex<float> *b, INTEGER ldb,
                            INTEGER *info);


INTEGER magma_cheevd_m( INTEGER nrgpu, 
                            char jobz, char uplo,
                            INTEGER n,
                            std::complex<float> *a, INTEGER lda,
                            float *w,
                            std::complex<float> *work, INTEGER lwork,
                            float *rwork, INTEGER lrwork,
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_chegvd_m( INTEGER nrgpu,
                            INTEGER itype, char jobz, char uplo,
                            INTEGER n,
                            std::complex<float> *a, INTEGER lda,
                            std::complex<float> *b, INTEGER ldb,
                            float *w,
                            std::complex<float> *work, INTEGER lwork,
                            float *rwork, INTEGER lrwork,
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_cheevdx_m( INTEGER nrgpu, 
                             char jobz, char range, char uplo,
                             INTEGER n,
                             std::complex<float> *a, INTEGER lda,
                             float vl, float vu, INTEGER il, INTEGER iu,
                             INTEGER *m, float *w,
                             std::complex<float> *work, INTEGER lwork,
                             float *rwork, INTEGER lrwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_chegvdx_m( INTEGER nrgpu,
                             INTEGER itype, char jobz, char range, char uplo,
                             INTEGER n,
                             std::complex<float> *a, INTEGER lda,
                             std::complex<float> *b, INTEGER ldb,
                             float vl, float vu, INTEGER il, INTEGER iu,
                             INTEGER *m, float *w,
                             std::complex<float> *work, INTEGER lwork,
                             float *rwork, INTEGER lrwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_cheevdx_2stage_m( INTEGER nrgpu, 
                                    char jobz, char range, char uplo,
                                    INTEGER n,
                                    std::complex<float> *a, INTEGER lda,
                                    float vl, float vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, float *w,
                                    std::complex<float> *work, INTEGER lwork,
                                    float *rwork, INTEGER lrwork,
                                    INTEGER *iwork, INTEGER liwork,
                                    INTEGER *info);
INTEGER magma_chegvdx_2stage_m( INTEGER nrgpu, 
                                    INTEGER itype, char jobz, char range, char uplo, 
                                    INTEGER n,
                                    std::complex<float> *a, INTEGER lda, 
                                    std::complex<float> *b, INTEGER ldb,
                                    float vl, float vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, float *w, 
                                    std::complex<float> *work, INTEGER lwork, 
                                    float *rwork, INTEGER lrwork, 
                                    INTEGER *iwork, INTEGER liwork, 
                                    INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
INTEGER magma_cgels_gpu(  char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<float> *dA,    INTEGER ldda,
                              std::complex<float> *dB,    INTEGER lddb,
                              std::complex<float> *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_cgels3_gpu( char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<float> *dA,    INTEGER ldda,
                              std::complex<float> *dB,    INTEGER lddb,
                              std::complex<float> *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_cgelqf_gpu( INTEGER m, INTEGER n,
                              std::complex<float> *dA,    INTEGER ldda,   std::complex<float> *tau,
                              std::complex<float> *work, INTEGER lwork, INTEGER *info);

INTEGER magma_cgeqr2x_gpu(
    INTEGER *m, INTEGER *n, std::complex<float> *dA,
    INTEGER *ldda, std::complex<float> *dtau,
    std::complex<float> *dT, std::complex<float> *ddA,
    float *dwork, INTEGER *info);

INTEGER magma_cgeqr2x2_gpu(
    INTEGER *m, INTEGER *n, std::complex<float> *dA,
    INTEGER *ldda, std::complex<float> *dtau,
    std::complex<float> *dT, std::complex<float> *ddA,
    float *dwork, INTEGER *info);

INTEGER magma_cgeqr2x3_gpu(
    INTEGER *m, INTEGER *n, std::complex<float> *dA,
    INTEGER *ldda, std::complex<float> *dtau,
    std::complex<float> *dT, std::complex<float> *ddA,
    float *dwork, INTEGER *info);

INTEGER magma_cgeqr2x4_gpu(
    INTEGER *m, INTEGER *n, std::complex<float> *dA,
    INTEGER *ldda, std::complex<float> *dtau,
    std::complex<float> *dT, std::complex<float> *ddA,
    float *dwork, INTEGER *info, magma_queue_t stream);

INTEGER magma_cgeqrf_gpu( INTEGER m, INTEGER n,
                              std::complex<float> *dA,  INTEGER ldda,
                              std::complex<float> *tau, std::complex<float> *dT,
                              INTEGER *info);
INTEGER magma_cgeqrf2_gpu(INTEGER m, INTEGER n,
                              std::complex<float> *dA,  INTEGER ldda,
                              std::complex<float> *tau, INTEGER *info);
INTEGER magma_cgeqrf2_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                               std::complex<float> **dlA, INTEGER ldda,
                               std::complex<float> *tau, INTEGER *info );
INTEGER magma_cgeqrf3_gpu(INTEGER m, INTEGER n,
                              std::complex<float> *dA,  INTEGER ldda,
                              std::complex<float> *tau, std::complex<float> *dT,
                              INTEGER *info);
INTEGER magma_cgeqr2_gpu( INTEGER m, INTEGER n,
                              std::complex<float> *dA,  INTEGER lda,
                              std::complex<float> *tau, float *work,
                              INTEGER *info);
INTEGER magma_cgeqrs_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<float> *dA,     INTEGER ldda,
                              std::complex<float> *tau,   std::complex<float> *dT,
                              std::complex<float> *dB,    INTEGER lddb,
                              std::complex<float> *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_cgeqrs3_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<float> *dA,     INTEGER ldda,
                              std::complex<float> *tau,   std::complex<float> *dT,
                              std::complex<float> *dB,    INTEGER lddb,
                              std::complex<float> *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_cgessm_gpu( char storev, INTEGER m, INTEGER n, INTEGER k, INTEGER ib,
                              INTEGER *ipiv,
                              std::complex<float> *dL1, INTEGER lddl1,
                              std::complex<float> *dL,  INTEGER lddl,
                              std::complex<float> *dA,  INTEGER ldda,
                              INTEGER *info);
INTEGER magma_cgesv_gpu(  INTEGER n, INTEGER nrhs,
                              std::complex<float> *dA, INTEGER ldda, INTEGER *ipiv,
                              std::complex<float> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_cgetf2_gpu( INTEGER m, INTEGER n,
                              std::complex<float> *dA, INTEGER lda, INTEGER *ipiv,
                              INTEGER* info );
INTEGER magma_cgetrf_incpiv_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib,
                              std::complex<float> *hA, INTEGER ldha, std::complex<float> *dA, INTEGER ldda,
                              std::complex<float> *hL, INTEGER ldhl, std::complex<float> *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              std::complex<float> *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_cgetrf_gpu( INTEGER m, INTEGER n,
                              std::complex<float> *dA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_cgetrf_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                              std::complex<float> **d_lA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_cgetrf_m(INTEGER num_gpus0, INTEGER m, INTEGER n, std::complex<float> *a, INTEGER lda,
                           INTEGER *ipiv, INTEGER *info);
INTEGER magma_cgetrf_piv(INTEGER m, INTEGER n, INTEGER NB,
                             std::complex<float> *a, INTEGER lda, INTEGER *ipiv,
                             INTEGER *info);
INTEGER magma_cgetrf2_mgpu(INTEGER num_gpus,
                               INTEGER m, INTEGER n, INTEGER nb, INTEGER offset,
                               std::complex<float> *d_lAT[], INTEGER lddat, INTEGER *ipiv,
                               std::complex<float> *d_lAP[], std::complex<float> *a, INTEGER lda,
                               magma_queue_t streaml[][2], INTEGER *info);
INTEGER
      magma_cgetrf_nopiv_gpu( INTEGER m, INTEGER n,
                              std::complex<float> *dA, INTEGER ldda,
                              INTEGER *info);
INTEGER magma_cgetri_gpu( INTEGER n,
                              std::complex<float> *dA, INTEGER ldda, const INTEGER *ipiv,
                              std::complex<float> *dwork, INTEGER lwork, INTEGER *info);
INTEGER magma_cgetrs_gpu( char trans, INTEGER n, INTEGER nrhs,
                              const std::complex<float> *dA, INTEGER ldda, const INTEGER *ipiv,
                              std::complex<float> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_clabrd_gpu( INTEGER m, INTEGER n, INTEGER nb,
                              std::complex<float> *a, INTEGER lda, std::complex<float> *da, INTEGER ldda,
                              float *d, float *e, std::complex<float> *tauq, std::complex<float> *taup,
                              std::complex<float> *x, INTEGER ldx, std::complex<float> *dx, INTEGER lddx,
                              std::complex<float> *y, INTEGER ldy, std::complex<float> *dy, INTEGER lddy);

INTEGER magma_claqps_gpu(
    INTEGER m, INTEGER n, INTEGER offset,
    INTEGER nb, INTEGER *kb,
    std::complex<float> *A,  INTEGER lda,
    INTEGER *jpvt, std::complex<float> *tau,
    float *vn1, float *vn2,
    std::complex<float> *auxv,
    std::complex<float> *dF, INTEGER lddf);

INTEGER magma_claqps2_gpu(
    INTEGER m, INTEGER n, INTEGER offset,
    INTEGER nb, INTEGER *kb,
    std::complex<float> *A,  INTEGER lda,
    INTEGER *jpvt, std::complex<float> *tau,
    float *vn1, float *vn2,
    std::complex<float> *auxv,
    std::complex<float> *dF, INTEGER lddf);

INTEGER magma_claqps3_gpu(
    INTEGER m, INTEGER n, INTEGER offset,
    INTEGER nb, INTEGER *kb,
    std::complex<float> *A,  INTEGER lda,
    INTEGER *jpvt, std::complex<float> *tau,
    float *vn1, float *vn2,
    std::complex<float> *auxv,
    std::complex<float> *dF, INTEGER lddf);

INTEGER magma_clarf_gpu(  INTEGER m, INTEGER n, std::complex<float> *v, std::complex<float> *tau,
                              std::complex<float> *c, INTEGER ldc, float *xnorm);
INTEGER magma_clarfb_gpu( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const std::complex<float> *dv, INTEGER ldv,
                              const std::complex<float> *dt, INTEGER ldt,
                              std::complex<float> *dc,       INTEGER ldc,
                              std::complex<float> *dwork,    INTEGER ldwork );
INTEGER magma_clarfb2_gpu(INTEGER m, INTEGER n, INTEGER k,
                              const std::complex<float> *dV,    INTEGER ldv,
                              const std::complex<float> *dT,    INTEGER ldt,
                              std::complex<float> *dC,          INTEGER ldc,
                              std::complex<float> *dwork,       INTEGER ldwork );
INTEGER magma_clarfb_gpu_gemm( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const std::complex<float> *dv, INTEGER ldv,
                              const std::complex<float> *dt, INTEGER ldt,
                              std::complex<float> *dc,       INTEGER ldc,
                              std::complex<float> *dwork,    INTEGER ldwork,
                              std::complex<float> *dworkvt,  INTEGER ldworkvt);
INTEGER magma_clarfg_gpu( INTEGER n, std::complex<float> *dx0, std::complex<float> *dx,
                              std::complex<float> *dtau, float *dxnorm, std::complex<float> *dAkk);
INTEGER magma_cposv_gpu(  char uplo, INTEGER n, INTEGER nrhs,
                              std::complex<float> *dA, INTEGER ldda,
                              std::complex<float> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_cpotf2_gpu( magma_uplo_t uplo, INTEGER n, 
                              std::complex<float> *dA, INTEGER lda,
                              INTEGER *info );
INTEGER magma_cpotrf_gpu( char uplo,  INTEGER n,
                              std::complex<float> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_cpotrf_mgpu(INTEGER ngpu, char uplo, INTEGER n,
                              std::complex<float> **d_lA, INTEGER ldda, INTEGER *info);
INTEGER magma_cpotrf3_mgpu(INTEGER num_gpus, char uplo, INTEGER m, INTEGER n,
                               INTEGER off_i, INTEGER off_j, INTEGER nb,
                               std::complex<float> *d_lA[],  INTEGER ldda,
                               std::complex<float> *d_lP[],  INTEGER lddp,
                               std::complex<float> *a,      INTEGER lda,   INTEGER h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               INTEGER *info );
INTEGER magma_cpotri_gpu( char uplo,  INTEGER n,
                              std::complex<float> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_clauum_gpu( char uplo,  INTEGER n,
                              std::complex<float> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_ctrtri_gpu( char uplo,  char diag, INTEGER n,
                              std::complex<float> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_chetrd_gpu( char uplo, INTEGER n,
                              std::complex<float> *da, INTEGER ldda,
                              float *d, float *e, std::complex<float> *tau,
                              std::complex<float> *wa,  INTEGER ldwa,
                              std::complex<float> *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_chetrd2_gpu(char uplo, INTEGER n,
                              std::complex<float> *da, INTEGER ldda,
                              float *d, float *e, std::complex<float> *tau,
                              std::complex<float> *wa,  INTEGER ldwa,
                              std::complex<float> *work, INTEGER lwork,
                              std::complex<float> *dwork, INTEGER ldwork,
                              INTEGER *info);

float magma_clatrd_mgpu(INTEGER num_gpus, char uplo,
			  INTEGER n0, INTEGER n, INTEGER nb, INTEGER nb0,
			  std::complex<float> *a,  INTEGER lda,
			  float *e, std::complex<float> *tau,
			  std::complex<float> *w,   INTEGER ldw,
			  std::complex<float> **da, INTEGER ldda, INTEGER offset,
			  std::complex<float> **dw, INTEGER lddw,
			  std::complex<float> *dwork[MagmaMaxGPUs], INTEGER ldwork,
			  INTEGER k,
			  std::complex<float>  *dx[MagmaMaxGPUs], std::complex<float> *dy[MagmaMaxGPUs],
			  std::complex<float> *work,
			  magma_queue_t stream[][10],
			  float *times );

INTEGER magma_chetrd_mgpu(INTEGER num_gpus, INTEGER k, char uplo, INTEGER n,
                              std::complex<float> *a, INTEGER lda,
                              float *d, float *e, std::complex<float> *tau,
                              std::complex<float> *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_chetrd_hb2st(INTEGER threads, char uplo, 
                              INTEGER n, INTEGER nb, INTEGER Vblksiz,
                              std::complex<float> *A, INTEGER lda,
                              float *D, float *E,
                              std::complex<float> *V, INTEGER ldv,
                              std::complex<float> *TAU, INTEGER compT,
                              std::complex<float> *T, INTEGER ldt);
INTEGER magma_chetrd_he2hb(char uplo, INTEGER n, INTEGER NB,
                              std::complex<float> *a, INTEGER lda,
                              std::complex<float> *tau, std::complex<float> *work, INTEGER lwork,
                              std::complex<float> *dT, INTEGER threads,
                              INTEGER *info);
INTEGER magma_chetrd_he2hb_mgpu( char uplo, INTEGER n, INTEGER nb,
                              std::complex<float> *a, INTEGER lda,
                              std::complex<float> *tau,
                              std::complex<float> *work, INTEGER lwork,
                              std::complex<float> *dAmgpu[], INTEGER ldda,
                              std::complex<float> *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_chetrd_he2hb_mgpu_spec( char uplo, INTEGER n, INTEGER nb,
                              std::complex<float> *a, INTEGER lda,
                              std::complex<float> *tau,
                              std::complex<float> *work, INTEGER lwork,
                              std::complex<float> *dAmgpu[], INTEGER ldda,
                              std::complex<float> *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_cpotrs_gpu( char uplo,  INTEGER n, INTEGER nrhs,
                              std::complex<float> *dA, INTEGER ldda,
                              std::complex<float> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_cssssm_gpu( char storev, INTEGER m1, INTEGER n1,
                              INTEGER m2, INTEGER n2, INTEGER k, INTEGER ib,
                              std::complex<float> *dA1, INTEGER ldda1,
                              std::complex<float> *dA2, INTEGER ldda2,
                              std::complex<float> *dL1, INTEGER lddl1,
                              std::complex<float> *dL2, INTEGER lddl2,
                              INTEGER *IPIV, INTEGER *info);
INTEGER magma_ctstrf_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib, INTEGER nb,
                              std::complex<float> *hU, INTEGER ldhu, std::complex<float> *dU, INTEGER lddu,
                              std::complex<float> *hA, INTEGER ldha, std::complex<float> *dA, INTEGER ldda,
                              std::complex<float> *hL, INTEGER ldhl, std::complex<float> *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              std::complex<float> *hwork, INTEGER ldhwork,
                              std::complex<float> *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_cungqr_gpu( INTEGER m, INTEGER n, INTEGER k,
                              std::complex<float> *da, INTEGER ldda,
                              std::complex<float> *tau, std::complex<float> *dwork,
                              INTEGER nb, INTEGER *info );
INTEGER magma_cunmql2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              std::complex<float> *da, INTEGER ldda,
                              std::complex<float> *tau,
                              std::complex<float> *dc, INTEGER lddc,
                              std::complex<float> *wa, INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_cunmqr_gpu( char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              std::complex<float> *dA,    INTEGER ldda, std::complex<float> *tau,
                              std::complex<float> *dC,    INTEGER lddc,
                              std::complex<float> *hwork, INTEGER lwork,
                              std::complex<float> *dT,    INTEGER nb, INTEGER *info);
INTEGER magma_cunmqr2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              std::complex<float> *da,   INTEGER ldda,
                              std::complex<float> *tau,
                              std::complex<float> *dc,    INTEGER lddc,
                              std::complex<float> *wa,    INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_cunmtr_gpu( char side, char uplo, char trans,
                              INTEGER m, INTEGER n,
                              std::complex<float> *da,    INTEGER ldda,
                              std::complex<float> *tau,
                              std::complex<float> *dc,    INTEGER lddc,
                              std::complex<float> *wa,    INTEGER ldwa,
                              INTEGER *info);


INTEGER magma_cgeqp3_gpu( INTEGER m, INTEGER n,
                              std::complex<float> *A, INTEGER lda,
                              INTEGER *jpvt, std::complex<float> *tau,
                              std::complex<float> *work, INTEGER lwork,
                              float *rwork, INTEGER *info );
INTEGER magma_cheevd_gpu( char jobz, char uplo,
                              INTEGER n,
                              std::complex<float> *da, INTEGER ldda,
                              float *w,
                              std::complex<float> *wa,  INTEGER ldwa,
                              std::complex<float> *work, INTEGER lwork,
                              float *rwork, INTEGER lrwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);
INTEGER magma_cheevdx_gpu(char jobz, char range, char uplo,
                              INTEGER n, std::complex<float> *da,
                              INTEGER ldda, float vl, float vu,
                              INTEGER il, INTEGER iu,
                              INTEGER *m, float *w,
                              std::complex<float> *wa,  INTEGER ldwa,
                              std::complex<float> *work, INTEGER lwork,
                              float *rwork, INTEGER lrwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);
INTEGER magma_cheevx_gpu( char jobz, char range, char uplo, INTEGER n,
                              std::complex<float> *da, INTEGER ldda, float vl,
                              float vu, INTEGER il, INTEGER iu,
                              float abstol, INTEGER *m,
                              float *w, std::complex<float> *dz, INTEGER lddz,
                              std::complex<float> *wa, INTEGER ldwa,
                              std::complex<float> *wz, INTEGER ldwz,
                              std::complex<float> *work, INTEGER lwork,
                              float *rwork, INTEGER *iwork,
                              INTEGER *ifail, INTEGER *info);
INTEGER magma_cheevr_gpu( char jobz, char range, char uplo, INTEGER n,
                              std::complex<float> *da, INTEGER ldda, float vl, float vu,
                              INTEGER il, INTEGER iu, float abstol, INTEGER *m,
                              float *w, std::complex<float> *dz, INTEGER lddz,
                              INTEGER *isuppz,
                              std::complex<float> *wa, INTEGER ldwa,
                              std::complex<float> *wz, INTEGER ldwz,
                              std::complex<float> *work, INTEGER lwork,
                              float *rwork, INTEGER lrwork, INTEGER *iwork,
                              INTEGER liwork, INTEGER *info);

INTEGER magma_chegst_gpu(INTEGER itype, char uplo, INTEGER n,
                             std::complex<float> *da, INTEGER ldda,
                             std::complex<float> *db, INTEGER lddb, INTEGER *info);




/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
INTEGER magma_get_zpotrf_nb( INTEGER m );
INTEGER magma_get_zgetrf_nb( INTEGER m );
INTEGER magma_get_zgetri_nb( INTEGER m );
INTEGER magma_get_zgeqp3_nb( INTEGER m );
INTEGER magma_get_zgeqrf_nb( INTEGER m );
INTEGER magma_get_zgeqlf_nb( INTEGER m );
INTEGER magma_get_zgehrd_nb( INTEGER m );
INTEGER magma_get_zhetrd_nb( INTEGER m );
INTEGER magma_get_zgelqf_nb( INTEGER m );
INTEGER magma_get_zgebrd_nb( INTEGER m );
INTEGER magma_get_zhegst_nb( INTEGER m );
INTEGER magma_get_zgesvd_nb( INTEGER m );
INTEGER magma_get_zhegst_nb_m( INTEGER m );
INTEGER magma_get_zbulge_nb( INTEGER m, INTEGER nbthreads );
INTEGER magma_get_zbulge_nb_mgpu( INTEGER m );
INTEGER magma_zbulge_get_Vblksiz( INTEGER m, INTEGER nb, INTEGER nbthreads );
INTEGER magma_get_zbulge_gcperf();
INTEGER magma_get_smlsize_divideconquer();
/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/
void magma_smove_eig(char range, INTEGER n, float *w, INTEGER *il,
                          INTEGER *iu, float vl, float vu, INTEGER *m);
INTEGER magma_zgebrd( INTEGER m, INTEGER n, std::complex<double> *A,
                          INTEGER lda, float *d, float *e,
                          std::complex<double> *tauq,  std::complex<double> *taup,
                          std::complex<double> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_zgehrd2(INTEGER n, INTEGER ilo, INTEGER ihi,
                          std::complex<double> *A, INTEGER lda, std::complex<double> *tau,
                          std::complex<double> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_zgehrd( INTEGER n, INTEGER ilo, INTEGER ihi,
                          std::complex<double> *A, INTEGER lda, std::complex<double> *tau,
                          std::complex<double> *work, INTEGER lwork,
                          std::complex<double> *dT, INTEGER *info);
INTEGER magma_zgelqf( INTEGER m, INTEGER n,
                          std::complex<double> *A,    INTEGER lda,   std::complex<double> *tau,
                          std::complex<double> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_zgeqlf( INTEGER m, INTEGER n,
                          std::complex<double> *A,    INTEGER lda,   std::complex<double> *tau,
                          std::complex<double> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_zgeqrf( INTEGER m, INTEGER n, std::complex<double> *A,
                          INTEGER lda, std::complex<double> *tau, std::complex<double> *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_zgeqrf4(INTEGER num_gpus, INTEGER m, INTEGER n,
                          std::complex<double> *a,    INTEGER lda, std::complex<double> *tau,
                          std::complex<double> *work, INTEGER lwork, INTEGER *info );
INTEGER magma_zgeqrf_ooc( INTEGER m, INTEGER n, std::complex<double> *A,
                          INTEGER lda, std::complex<double> *tau, std::complex<double> *work,
                          INTEGER lwork, INTEGER *info);
INTEGER magma_zgesv ( INTEGER n, INTEGER nrhs,
                          std::complex<double> *A, INTEGER lda, INTEGER *ipiv,
                          std::complex<double> *B, INTEGER ldb, INTEGER *info);
INTEGER magma_zgetrf( INTEGER m, INTEGER n, std::complex<double> *A,
                          INTEGER lda, INTEGER *ipiv,
                          INTEGER *info);
INTEGER magma_zgetrf2(INTEGER m, INTEGER n, std::complex<double> *a,
                          INTEGER lda, INTEGER *ipiv, INTEGER *info);

INTEGER magma_zlaqps( INTEGER m, INTEGER n, INTEGER offset,
                          INTEGER nb, INTEGER *kb,
                          std::complex<double> *A,  INTEGER lda,
                          std::complex<double> *dA, INTEGER ldda,
                          INTEGER *jpvt, std::complex<double> *tau, float *vn1, float *vn2,
                          std::complex<double> *auxv,
                          std::complex<double> *F,  INTEGER ldf,
                          std::complex<double> *dF, INTEGER lddf );
void        magma_zlarfg( INTEGER n, std::complex<double> *alpha, std::complex<double> *x,
                          INTEGER incx, std::complex<double> *tau);
INTEGER magma_zlatrd( char uplo, INTEGER n, INTEGER nb, std::complex<double> *a,
                          INTEGER lda, float *e, std::complex<double> *tau,
                          std::complex<double> *w, INTEGER ldw,
                          std::complex<double> *da, INTEGER ldda,
                          std::complex<double> *dw, INTEGER lddw);
INTEGER magma_zlatrd2(char uplo, INTEGER n, INTEGER nb,
                          std::complex<double> *a,  INTEGER lda,
                          float *e, std::complex<double> *tau,
                          std::complex<double> *w,  INTEGER ldw,
                          std::complex<double> *da, INTEGER ldda,
                          std::complex<double> *dw, INTEGER lddw,
                          std::complex<double> *dwork, INTEGER ldwork);
INTEGER magma_zlahr2( INTEGER m, INTEGER n, INTEGER nb,
                          std::complex<double> *da, std::complex<double> *dv, std::complex<double> *a,
                          INTEGER lda, std::complex<double> *tau, std::complex<double> *t,
                          INTEGER ldt, std::complex<double> *y, INTEGER ldy);
INTEGER magma_zlahru( INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *da, std::complex<double> *y,
                          std::complex<double> *v, std::complex<double> *t,
                          std::complex<double> *dwork);
INTEGER magma_zposv ( char uplo, INTEGER n, INTEGER nrhs,
                          std::complex<double> *A, INTEGER lda,
                          std::complex<double> *B, INTEGER ldb, INTEGER *info);
INTEGER magma_zpotrf( char uplo, INTEGER n, std::complex<double> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_zpotri( char uplo, INTEGER n, std::complex<double> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_zlauum( char uplo, INTEGER n, std::complex<double> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_ztrtri( char uplo, char diag, INTEGER n, std::complex<double> *A,
                          INTEGER lda, INTEGER *info);
INTEGER magma_zhetrd( char uplo, INTEGER n, std::complex<double> *A,
                          INTEGER lda, float *d, float *e,
                          std::complex<double> *tau, std::complex<double> *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_zungqr( INTEGER m, INTEGER n, INTEGER k,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *tau, std::complex<double> *dT,
                          INTEGER nb, INTEGER *info );
INTEGER magma_zungqr2(INTEGER m, INTEGER n, INTEGER k,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *tau, INTEGER *info );
INTEGER magma_zunmql( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *tau,
                          std::complex<double> *c, INTEGER ldc,
                          std::complex<double> *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_zunmqr( char side, char trans,
                          INTEGER m, INTEGER n, INTEGER k,
                          std::complex<double> *a, INTEGER lda, std::complex<double> *tau,
                          std::complex<double> *c, INTEGER ldc,
                          std::complex<double> *work, INTEGER lwork, INTEGER *info);
INTEGER magma_zunmtr( char side, char uplo, char trans,
                          INTEGER m, INTEGER n,
                          std::complex<double> *a,    INTEGER lda,
                          std::complex<double> *tau,
                          std::complex<double> *c,    INTEGER ldc,
                          std::complex<double> *work, INTEGER lwork,
                          INTEGER *info);
INTEGER magma_zunghr( INTEGER n, INTEGER ilo, INTEGER ihi,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *tau,
                          std::complex<double> *dT, INTEGER nb,
                          INTEGER *info);


INTEGER  magma_zgeev( char jobvl, char jobvr, INTEGER n,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *w,
                          std::complex<double> *vl, INTEGER ldvl,
                          std::complex<double> *vr, INTEGER ldvr,
                          std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER *info);
INTEGER magma_zgeqp3( INTEGER m, INTEGER n,
                          std::complex<double> *a, INTEGER lda,
                          INTEGER *jpvt, std::complex<double> *tau,
                          std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER *info);
INTEGER magma_zgesvd( char jobu, char jobvt, INTEGER m, INTEGER n,
                          std::complex<double> *a,    INTEGER lda, float *s,
                          std::complex<double> *u,    INTEGER ldu,
                          std::complex<double> *vt,   INTEGER ldvt,
                          std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER *info );
INTEGER magma_zheevd( char jobz, char uplo, INTEGER n,
                          std::complex<double> *a, INTEGER lda, float *w,
                          std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_zheevdx(char jobz, char range, char uplo, INTEGER n,
                          std::complex<double> *a, INTEGER lda,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, std::complex<double> *work,
                          INTEGER lwork, float *rwork, INTEGER lrwork,
                          INTEGER *iwork, INTEGER liwork, INTEGER *info);
INTEGER magma_zheevdx_2stage(char jobz, char range, char uplo,
                          INTEGER n,
                          std::complex<double> *a, INTEGER lda,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w,
                          std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork,
                          INTEGER *iwork, INTEGER liwork,
                          INTEGER *info);
INTEGER magma_zheevx( char jobz, char range, char uplo, INTEGER n,
                          std::complex<double> *a, INTEGER lda, float vl, float vu,
                          INTEGER il, INTEGER iu, float abstol, INTEGER *m,
                          float *w, std::complex<double> *z, INTEGER ldz,
                          std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER *iwork,
                          INTEGER *ifail, INTEGER *info);
INTEGER magma_zheevr( char jobz, char range, char uplo, INTEGER n,
                          std::complex<double> *a, INTEGER lda, float vl, float vu,
                          INTEGER il, INTEGER iu, float abstol, INTEGER *m,
                          float *w, std::complex<double> *z, INTEGER ldz,
                          INTEGER *isuppz,
                          std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_zhegvd( INTEGER itype, char jobz, char uplo, INTEGER n,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *b, INTEGER ldb,
                          float *w, std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_zhegvdx(INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, std::complex<double> *a, INTEGER lda,
                          std::complex<double> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, std::complex<double> *work,
                          INTEGER lwork, float *rwork,
                          INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_zhegvdx_2stage(INTEGER itype, char jobz, char range, char uplo, INTEGER n,
                          std::complex<double> *a, INTEGER lda, 
                          std::complex<double> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          INTEGER *m, float *w, 
                          std::complex<double> *work, INTEGER lwork, 
                          float *rwork, INTEGER lrwork, 
                          INTEGER *iwork, INTEGER liwork, 
                          INTEGER *info);
INTEGER magma_zhegvx( INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, std::complex<double> *a, INTEGER lda,
                          std::complex<double> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          float abstol, INTEGER *m, float *w,
                          std::complex<double> *z, INTEGER ldz,
                          std::complex<double> *work, INTEGER lwork, float *rwork,
                          INTEGER *iwork, INTEGER *ifail, INTEGER *info);
INTEGER magma_zhegvr( INTEGER itype, char jobz, char range, char uplo,
                          INTEGER n, std::complex<double> *a, INTEGER lda,
                          std::complex<double> *b, INTEGER ldb,
                          float vl, float vu, INTEGER il, INTEGER iu,
                          float abstol, INTEGER *m, float *w,
                          std::complex<double> *z, INTEGER ldz,
                          INTEGER *isuppz, std::complex<double> *work, INTEGER lwork,
                          float *rwork, INTEGER lrwork, INTEGER *iwork,
                          INTEGER liwork, INTEGER *info);
INTEGER magma_zstedx( char range, INTEGER n, float vl, float vu,
                          INTEGER il, INTEGER iu, float *D, float *E,
                          std::complex<double> *Z, INTEGER ldz,
                          float *rwork, INTEGER ldrwork, INTEGER *iwork,
                          INTEGER liwork, float *dwork, INTEGER *info);



INTEGER magma_zhegst( INTEGER itype, char uplo, INTEGER n,
                          std::complex<double> *a, INTEGER lda,
                          std::complex<double> *b, INTEGER ldb, INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
INTEGER magma_zlahr2_m(
    INTEGER n, INTEGER k, INTEGER nb,
    std::complex<double> *A, INTEGER lda,
    std::complex<double> *tau,
    std::complex<double> *T, INTEGER ldt,
    std::complex<double> *Y, INTEGER ldy,
    struct cgehrd_data *data );

INTEGER magma_zlahru_m(
    INTEGER n, INTEGER ihi, INTEGER k, INTEGER nb,
    std::complex<double> *A, INTEGER lda,
    struct cgehrd_data *data );

INTEGER magma_zgeev_m(
    char jobvl, char jobvr, INTEGER n,
    std::complex<double> *A, INTEGER lda,
    std::complex<double> *W,
    std::complex<double> *vl, INTEGER ldvl,
    std::complex<double> *vr, INTEGER ldvr,
    std::complex<double> *work, INTEGER lwork,
    float *rwork,
    INTEGER *info );


INTEGER magma_zgehrd_m(
    INTEGER n, INTEGER ilo, INTEGER ihi,
    std::complex<double> *A, INTEGER lda,
    std::complex<double> *tau,
    std::complex<double> *work, INTEGER lwork,
    std::complex<double> *T,
    INTEGER *info );

INTEGER magma_zunghr_m(
    INTEGER n, INTEGER ilo, INTEGER ihi,
    std::complex<double> *A, INTEGER lda,
    std::complex<double> *tau,
    std::complex<double> *T, INTEGER nb,
    INTEGER *info );

INTEGER magma_zungqr_m(
    INTEGER m, INTEGER n, INTEGER k,
    std::complex<double> *A, INTEGER lda,
    std::complex<double> *tau,
    std::complex<double> *T, INTEGER nb,
    INTEGER *info );

INTEGER magma_zpotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            std::complex<double> *A, INTEGER lda,
                            INTEGER *info);
INTEGER magma_zpotrf_m( INTEGER num_gpus,
                            char uplo, INTEGER n,
                            std::complex<double> *a, INTEGER lda, 
                            INTEGER *info);
INTEGER magma_zstedx_m( INTEGER nrgpu,
                            char range, INTEGER n, float vl, float vu,
                            INTEGER il, INTEGER iu, float *D, float *E,
                            std::complex<double> *Z, INTEGER ldz,
                            float *rwork, INTEGER ldrwork, INTEGER *iwork,
                            INTEGER liwork, INTEGER *info);
INTEGER magma_ztrsm_m ( INTEGER nrgpu,
                            char side, char uplo, char transa, char diag,
                            INTEGER m, INTEGER n, std::complex<double> alpha,
                            std::complex<double> *a, INTEGER lda, 
                            std::complex<double> *b, INTEGER ldb);
INTEGER magma_zunmqr_m( INTEGER nrgpu, char side, char trans,
                            INTEGER m, INTEGER n, INTEGER k,
                            std::complex<double> *a,    INTEGER lda,
                            std::complex<double> *tau,
                            std::complex<double> *c,    INTEGER ldc,
                            std::complex<double> *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_zunmtr_m( INTEGER nrgpu,
                            char side, char uplo, char trans,
                            INTEGER m, INTEGER n,
                            std::complex<double> *a,    INTEGER lda,
                            std::complex<double> *tau,
                            std::complex<double> *c,    INTEGER ldc,
                            std::complex<double> *work, INTEGER lwork,
                            INTEGER *info);
INTEGER magma_zhegst_m( INTEGER nrgpu,
                            INTEGER itype, char uplo, INTEGER n,
                            std::complex<double> *a, INTEGER lda,
                            std::complex<double> *b, INTEGER ldb,
                            INTEGER *info);


INTEGER magma_zheevd_m( INTEGER nrgpu, 
                            char jobz, char uplo,
                            INTEGER n,
                            std::complex<double> *a, INTEGER lda,
                            float *w,
                            std::complex<double> *work, INTEGER lwork,
                            float *rwork, INTEGER lrwork,
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_zhegvd_m( INTEGER nrgpu,
                            INTEGER itype, char jobz, char uplo,
                            INTEGER n,
                            std::complex<double> *a, INTEGER lda,
                            std::complex<double> *b, INTEGER ldb,
                            float *w,
                            std::complex<double> *work, INTEGER lwork,
                            float *rwork, INTEGER lrwork,
                            INTEGER *iwork, INTEGER liwork,
                            INTEGER *info);
INTEGER magma_zheevdx_m( INTEGER nrgpu, 
                             char jobz, char range, char uplo,
                             INTEGER n,
                             std::complex<double> *a, INTEGER lda,
                             float vl, float vu, INTEGER il, INTEGER iu,
                             INTEGER *m, float *w,
                             std::complex<double> *work, INTEGER lwork,
                             float *rwork, INTEGER lrwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_zhegvdx_m( INTEGER nrgpu,
                             INTEGER itype, char jobz, char range, char uplo,
                             INTEGER n,
                             std::complex<double> *a, INTEGER lda,
                             std::complex<double> *b, INTEGER ldb,
                             float vl, float vu, INTEGER il, INTEGER iu,
                             INTEGER *m, float *w,
                             std::complex<double> *work, INTEGER lwork,
                             float *rwork, INTEGER lrwork,
                             INTEGER *iwork, INTEGER liwork,
                             INTEGER *info);
INTEGER magma_zheevdx_2stage_m( INTEGER nrgpu, 
                                    char jobz, char range, char uplo,
                                    INTEGER n,
                                    std::complex<double> *a, INTEGER lda,
                                    float vl, float vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, float *w,
                                    std::complex<double> *work, INTEGER lwork,
                                    float *rwork, INTEGER lrwork,
                                    INTEGER *iwork, INTEGER liwork,
                                    INTEGER *info);
INTEGER magma_zhegvdx_2stage_m( INTEGER nrgpu, 
                                    INTEGER itype, char jobz, char range, char uplo, 
                                    INTEGER n,
                                    std::complex<double> *a, INTEGER lda, 
                                    std::complex<double> *b, INTEGER ldb,
                                    float vl, float vu, INTEGER il, INTEGER iu,
                                    INTEGER *m, float *w, 
                                    std::complex<double> *work, INTEGER lwork, 
                                    float *rwork, INTEGER lrwork, 
                                    INTEGER *iwork, INTEGER liwork, 
                                    INTEGER *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
INTEGER magma_zgels_gpu(  char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<double> *dA,    INTEGER ldda,
                              std::complex<double> *dB,    INTEGER lddb,
                              std::complex<double> *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_zgels3_gpu( char trans, INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<double> *dA,    INTEGER ldda,
                              std::complex<double> *dB,    INTEGER lddb,
                              std::complex<double> *hwork, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_zgelqf_gpu( INTEGER m, INTEGER n,
                              std::complex<double> *dA,    INTEGER ldda,   std::complex<double> *tau,
                              std::complex<double> *work, INTEGER lwork, INTEGER *info);

INTEGER magma_zgeqr2x_gpu(
    INTEGER *m, INTEGER *n, std::complex<double> *dA,
    INTEGER *ldda, std::complex<double> *dtau,
    std::complex<double> *dT, std::complex<double> *ddA,
    float *dwork, INTEGER *info);

INTEGER magma_zgeqr2x2_gpu(
    INTEGER *m, INTEGER *n, std::complex<double> *dA,
    INTEGER *ldda, std::complex<double> *dtau,
    std::complex<double> *dT, std::complex<double> *ddA,
    float *dwork, INTEGER *info);

INTEGER magma_zgeqr2x3_gpu(
    INTEGER *m, INTEGER *n, std::complex<double> *dA,
    INTEGER *ldda, std::complex<double> *dtau,
    std::complex<double> *dT, std::complex<double> *ddA,
    float *dwork, INTEGER *info);

INTEGER magma_zgeqr2x4_gpu(
    INTEGER *m, INTEGER *n, std::complex<double> *dA,
    INTEGER *ldda, std::complex<double> *dtau,
    std::complex<double> *dT, std::complex<double> *ddA,
    float *dwork, INTEGER *info, magma_queue_t stream);

INTEGER magma_zgeqrf_gpu( INTEGER m, INTEGER n,
                              std::complex<double> *dA,  INTEGER ldda,
                              std::complex<double> *tau, std::complex<double> *dT,
                              INTEGER *info);
INTEGER magma_zgeqrf2_gpu(INTEGER m, INTEGER n,
                              std::complex<double> *dA,  INTEGER ldda,
                              std::complex<double> *tau, INTEGER *info);
INTEGER magma_zgeqrf2_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                               std::complex<double> **dlA, INTEGER ldda,
                               std::complex<double> *tau, INTEGER *info );
INTEGER magma_zgeqrf3_gpu(INTEGER m, INTEGER n,
                              std::complex<double> *dA,  INTEGER ldda,
                              std::complex<double> *tau, std::complex<double> *dT,
                              INTEGER *info);
INTEGER magma_zgeqr2_gpu( INTEGER m, INTEGER n,
                              std::complex<double> *dA,  INTEGER lda,
                              std::complex<double> *tau, float *work,
                              INTEGER *info);
INTEGER magma_zgeqrs_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<double> *dA,     INTEGER ldda,
                              std::complex<double> *tau,   std::complex<double> *dT,
                              std::complex<double> *dB,    INTEGER lddb,
                              std::complex<double> *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_zgeqrs3_gpu( INTEGER m, INTEGER n, INTEGER nrhs,
                              std::complex<double> *dA,     INTEGER ldda,
                              std::complex<double> *tau,   std::complex<double> *dT,
                              std::complex<double> *dB,    INTEGER lddb,
                              std::complex<double> *hwork, INTEGER lhwork,
                              INTEGER *info);
INTEGER magma_zgessm_gpu( char storev, INTEGER m, INTEGER n, INTEGER k, INTEGER ib,
                              INTEGER *ipiv,
                              std::complex<double> *dL1, INTEGER lddl1,
                              std::complex<double> *dL,  INTEGER lddl,
                              std::complex<double> *dA,  INTEGER ldda,
                              INTEGER *info);
INTEGER magma_zgesv_gpu(  INTEGER n, INTEGER nrhs,
                              std::complex<double> *dA, INTEGER ldda, INTEGER *ipiv,
                              std::complex<double> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_zgetf2_gpu( INTEGER m, INTEGER n,
                              std::complex<double> *dA, INTEGER lda, INTEGER *ipiv,
                              INTEGER* info );
INTEGER magma_zgetrf_incpiv_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib,
                              std::complex<double> *hA, INTEGER ldha, std::complex<double> *dA, INTEGER ldda,
                              std::complex<double> *hL, INTEGER ldhl, std::complex<double> *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              std::complex<double> *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_zgetrf_gpu( INTEGER m, INTEGER n,
                              std::complex<double> *dA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_zgetrf_mgpu(INTEGER num_gpus, INTEGER m, INTEGER n,
                              std::complex<double> **d_lA, INTEGER ldda,
                              INTEGER *ipiv, INTEGER *info);
INTEGER magma_zgetrf_m(INTEGER num_gpus0, INTEGER m, INTEGER n, std::complex<double> *a, INTEGER lda,
                           INTEGER *ipiv, INTEGER *info);
INTEGER magma_zgetrf_piv(INTEGER m, INTEGER n, INTEGER NB,
                             std::complex<double> *a, INTEGER lda, INTEGER *ipiv,
                             INTEGER *info);
INTEGER magma_zgetrf2_mgpu(INTEGER num_gpus,
                               INTEGER m, INTEGER n, INTEGER nb, INTEGER offset,
                               std::complex<double> *d_lAT[], INTEGER lddat, INTEGER *ipiv,
                               std::complex<double> *d_lAP[], std::complex<double> *a, INTEGER lda,
                               magma_queue_t streaml[][2], INTEGER *info);
INTEGER
      magma_zgetrf_nopiv_gpu( INTEGER m, INTEGER n,
                              std::complex<double> *dA, INTEGER ldda,
                              INTEGER *info);
INTEGER magma_zgetri_gpu( INTEGER n,
                              std::complex<double> *dA, INTEGER ldda, const INTEGER *ipiv,
                              std::complex<double> *dwork, INTEGER lwork, INTEGER *info);
INTEGER magma_zgetrs_gpu( char trans, INTEGER n, INTEGER nrhs,
                              const std::complex<double> *dA, INTEGER ldda, const INTEGER *ipiv,
                              std::complex<double> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_zlabrd_gpu( INTEGER m, INTEGER n, INTEGER nb,
                              std::complex<double> *a, INTEGER lda, std::complex<double> *da, INTEGER ldda,
                              float *d, float *e, std::complex<double> *tauq, std::complex<double> *taup,
                              std::complex<double> *x, INTEGER ldx, std::complex<double> *dx, INTEGER lddx,
                              std::complex<double> *y, INTEGER ldy, std::complex<double> *dy, INTEGER lddy);

INTEGER magma_zlaqps_gpu(
    INTEGER m, INTEGER n, INTEGER offset,
    INTEGER nb, INTEGER *kb,
    std::complex<double> *A,  INTEGER lda,
    INTEGER *jpvt, std::complex<double> *tau,
    float *vn1, float *vn2,
    std::complex<double> *auxv,
    std::complex<double> *dF, INTEGER lddf);

INTEGER magma_zlaqps2_gpu(
    INTEGER m, INTEGER n, INTEGER offset,
    INTEGER nb, INTEGER *kb,
    std::complex<double> *A,  INTEGER lda,
    INTEGER *jpvt, std::complex<double> *tau,
    float *vn1, float *vn2,
    std::complex<double> *auxv,
    std::complex<double> *dF, INTEGER lddf);

INTEGER magma_zlaqps3_gpu(
    INTEGER m, INTEGER n, INTEGER offset,
    INTEGER nb, INTEGER *kb,
    std::complex<double> *A,  INTEGER lda,
    INTEGER *jpvt, std::complex<double> *tau,
    float *vn1, float *vn2,
    std::complex<double> *auxv,
    std::complex<double> *dF, INTEGER lddf);

INTEGER magma_zlarf_gpu(  INTEGER m, INTEGER n, std::complex<double> *v, std::complex<double> *tau,
                              std::complex<double> *c, INTEGER ldc, float *xnorm);
INTEGER magma_zlarfb_gpu( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const std::complex<double> *dv, INTEGER ldv,
                              const std::complex<double> *dt, INTEGER ldt,
                              std::complex<double> *dc,       INTEGER ldc,
                              std::complex<double> *dwork,    INTEGER ldwork );
INTEGER magma_zlarfb2_gpu(INTEGER m, INTEGER n, INTEGER k,
                              const std::complex<double> *dV,    INTEGER ldv,
                              const std::complex<double> *dT,    INTEGER ldt,
                              std::complex<double> *dC,          INTEGER ldc,
                              std::complex<double> *dwork,       INTEGER ldwork );
INTEGER magma_zlarfb_gpu_gemm( char side, char trans, char direct, char storev,
                              INTEGER m, INTEGER n, INTEGER k,
                              const std::complex<double> *dv, INTEGER ldv,
                              const std::complex<double> *dt, INTEGER ldt,
                              std::complex<double> *dc,       INTEGER ldc,
                              std::complex<double> *dwork,    INTEGER ldwork,
                              std::complex<double> *dworkvt,  INTEGER ldworkvt);
INTEGER magma_zlarfg_gpu( INTEGER n, std::complex<double> *dx0, std::complex<double> *dx,
                              std::complex<double> *dtau, float *dxnorm, std::complex<double> *dAkk);
INTEGER magma_zposv_gpu(  char uplo, INTEGER n, INTEGER nrhs,
                              std::complex<double> *dA, INTEGER ldda,
                              std::complex<double> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_zpotf2_gpu( magma_uplo_t uplo, INTEGER n, 
                              std::complex<double> *dA, INTEGER lda,
                              INTEGER *info );
INTEGER magma_zpotrf_gpu( char uplo,  INTEGER n,
                              std::complex<double> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_zpotrf_mgpu(INTEGER ngpu, char uplo, INTEGER n,
                              std::complex<double> **d_lA, INTEGER ldda, INTEGER *info);
INTEGER magma_zpotrf3_mgpu(INTEGER num_gpus, char uplo, INTEGER m, INTEGER n,
                               INTEGER off_i, INTEGER off_j, INTEGER nb,
                               std::complex<double> *d_lA[],  INTEGER ldda,
                               std::complex<double> *d_lP[],  INTEGER lddp,
                               std::complex<double> *a,      INTEGER lda,   INTEGER h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               INTEGER *info );
INTEGER magma_zpotri_gpu( char uplo,  INTEGER n,
                              std::complex<double> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_zlauum_gpu( char uplo,  INTEGER n,
                              std::complex<double> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_ztrtri_gpu( char uplo,  char diag, INTEGER n,
                              std::complex<double> *dA, INTEGER ldda, INTEGER *info);
INTEGER magma_zhetrd_gpu( char uplo, INTEGER n,
                              std::complex<double> *da, INTEGER ldda,
                              float *d, float *e, std::complex<double> *tau,
                              std::complex<double> *wa,  INTEGER ldwa,
                              std::complex<double> *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_zhetrd2_gpu(char uplo, INTEGER n,
                              std::complex<double> *da, INTEGER ldda,
                              float *d, float *e, std::complex<double> *tau,
                              std::complex<double> *wa,  INTEGER ldwa,
                              std::complex<double> *work, INTEGER lwork,
                              std::complex<double> *dwork, INTEGER ldwork,
                              INTEGER *info);

float magma_zlatrd_mgpu(INTEGER num_gpus, char uplo,
			  INTEGER n0, INTEGER n, INTEGER nb, INTEGER nb0,
			  std::complex<double> *a,  INTEGER lda,
			  float *e, std::complex<double> *tau,
			  std::complex<double> *w,   INTEGER ldw,
			  std::complex<double> **da, INTEGER ldda, INTEGER offset,
			  std::complex<double> **dw, INTEGER lddw,
			  std::complex<double> *dwork[MagmaMaxGPUs], INTEGER ldwork,
			  INTEGER k,
			  std::complex<double>  *dx[MagmaMaxGPUs], std::complex<double> *dy[MagmaMaxGPUs],
			  std::complex<double> *work,
			  magma_queue_t stream[][10],
			  float *times );

INTEGER magma_zhetrd_mgpu(INTEGER num_gpus, INTEGER k, char uplo, INTEGER n,
                              std::complex<double> *a, INTEGER lda,
                              float *d, float *e, std::complex<double> *tau,
                              std::complex<double> *work, INTEGER lwork,
                              INTEGER *info);
INTEGER magma_zhetrd_hb2st(INTEGER threads, char uplo, 
                              INTEGER n, INTEGER nb, INTEGER Vblksiz,
                              std::complex<double> *A, INTEGER lda,
                              float *D, float *E,
                              std::complex<double> *V, INTEGER ldv,
                              std::complex<double> *TAU, INTEGER compT,
                              std::complex<double> *T, INTEGER ldt);
INTEGER magma_zhetrd_he2hb(char uplo, INTEGER n, INTEGER NB,
                              std::complex<double> *a, INTEGER lda,
                              std::complex<double> *tau, std::complex<double> *work, INTEGER lwork,
                              std::complex<double> *dT, INTEGER threads,
                              INTEGER *info);
INTEGER magma_zhetrd_he2hb_mgpu( char uplo, INTEGER n, INTEGER nb,
                              std::complex<double> *a, INTEGER lda,
                              std::complex<double> *tau,
                              std::complex<double> *work, INTEGER lwork,
                              std::complex<double> *dAmgpu[], INTEGER ldda,
                              std::complex<double> *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_zhetrd_he2hb_mgpu_spec( char uplo, INTEGER n, INTEGER nb,
                              std::complex<double> *a, INTEGER lda,
                              std::complex<double> *tau,
                              std::complex<double> *work, INTEGER lwork,
                              std::complex<double> *dAmgpu[], INTEGER ldda,
                              std::complex<double> *dTmgpu[], INTEGER lddt,
                              INTEGER ngpu, INTEGER distblk,
                              magma_queue_t streams[][20], INTEGER nstream,
                              INTEGER threads, INTEGER *info);
INTEGER magma_zpotrs_gpu( char uplo,  INTEGER n, INTEGER nrhs,
                              std::complex<double> *dA, INTEGER ldda,
                              std::complex<double> *dB, INTEGER lddb, INTEGER *info);
INTEGER magma_zssssm_gpu( char storev, INTEGER m1, INTEGER n1,
                              INTEGER m2, INTEGER n2, INTEGER k, INTEGER ib,
                              std::complex<double> *dA1, INTEGER ldda1,
                              std::complex<double> *dA2, INTEGER ldda2,
                              std::complex<double> *dL1, INTEGER lddl1,
                              std::complex<double> *dL2, INTEGER lddl2,
                              INTEGER *IPIV, INTEGER *info);
INTEGER magma_ztstrf_gpu( char storev, INTEGER m, INTEGER n, INTEGER ib, INTEGER nb,
                              std::complex<double> *hU, INTEGER ldhu, std::complex<double> *dU, INTEGER lddu,
                              std::complex<double> *hA, INTEGER ldha, std::complex<double> *dA, INTEGER ldda,
                              std::complex<double> *hL, INTEGER ldhl, std::complex<double> *dL, INTEGER lddl,
                              INTEGER *ipiv,
                              std::complex<double> *hwork, INTEGER ldhwork,
                              std::complex<double> *dwork, INTEGER lddwork,
                              INTEGER *info);
INTEGER magma_zungqr_gpu( INTEGER m, INTEGER n, INTEGER k,
                              std::complex<double> *da, INTEGER ldda,
                              std::complex<double> *tau, std::complex<double> *dwork,
                              INTEGER nb, INTEGER *info );
INTEGER magma_zunmql2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              std::complex<double> *da, INTEGER ldda,
                              std::complex<double> *tau,
                              std::complex<double> *dc, INTEGER lddc,
                              std::complex<double> *wa, INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_zunmqr_gpu( char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              std::complex<double> *dA,    INTEGER ldda, std::complex<double> *tau,
                              std::complex<double> *dC,    INTEGER lddc,
                              std::complex<double> *hwork, INTEGER lwork,
                              std::complex<double> *dT,    INTEGER nb, INTEGER *info);
INTEGER magma_zunmqr2_gpu(char side, char trans,
                              INTEGER m, INTEGER n, INTEGER k,
                              std::complex<double> *da,   INTEGER ldda,
                              std::complex<double> *tau,
                              std::complex<double> *dc,    INTEGER lddc,
                              std::complex<double> *wa,    INTEGER ldwa,
                              INTEGER *info);
INTEGER magma_zunmtr_gpu( char side, char uplo, char trans,
                              INTEGER m, INTEGER n,
                              std::complex<double> *da,    INTEGER ldda,
                              std::complex<double> *tau,
                              std::complex<double> *dc,    INTEGER lddc,
                              std::complex<double> *wa,    INTEGER ldwa,
                              INTEGER *info);


INTEGER magma_zgeqp3_gpu( INTEGER m, INTEGER n,
                              std::complex<double> *A, INTEGER lda,
                              INTEGER *jpvt, std::complex<double> *tau,
                              std::complex<double> *work, INTEGER lwork,
                              float *rwork, INTEGER *info );
INTEGER magma_zheevd_gpu( char jobz, char uplo,
                              INTEGER n,
                              std::complex<double> *da, INTEGER ldda,
                              float *w,
                              std::complex<double> *wa,  INTEGER ldwa,
                              std::complex<double> *work, INTEGER lwork,
                              float *rwork, INTEGER lrwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);
INTEGER magma_zheevdx_gpu(char jobz, char range, char uplo,
                              INTEGER n, std::complex<double> *da,
                              INTEGER ldda, float vl, float vu,
                              INTEGER il, INTEGER iu,
                              INTEGER *m, float *w,
                              std::complex<double> *wa,  INTEGER ldwa,
                              std::complex<double> *work, INTEGER lwork,
                              float *rwork, INTEGER lrwork,
                              INTEGER *iwork, INTEGER liwork,
                              INTEGER *info);
INTEGER magma_zheevx_gpu( char jobz, char range, char uplo, INTEGER n,
                              std::complex<double> *da, INTEGER ldda, float vl,
                              float vu, INTEGER il, INTEGER iu,
                              float abstol, INTEGER *m,
                              float *w, std::complex<double> *dz, INTEGER lddz,
                              std::complex<double> *wa, INTEGER ldwa,
                              std::complex<double> *wz, INTEGER ldwz,
                              std::complex<double> *work, INTEGER lwork,
                              float *rwork, INTEGER *iwork,
                              INTEGER *ifail, INTEGER *info);
INTEGER magma_zheevr_gpu( char jobz, char range, char uplo, INTEGER n,
                              std::complex<double> *da, INTEGER ldda, float vl, float vu,
                              INTEGER il, INTEGER iu, float abstol, INTEGER *m,
                              float *w, std::complex<double> *dz, INTEGER lddz,
                              INTEGER *isuppz,
                              std::complex<double> *wa, INTEGER ldwa,
                              std::complex<double> *wz, INTEGER ldwz,
                              std::complex<double> *work, INTEGER lwork,
                              float *rwork, INTEGER lrwork, INTEGER *iwork,
                              INTEGER liwork, INTEGER *info);

INTEGER magma_zhegst_gpu(INTEGER itype, char uplo, INTEGER n,
                             std::complex<double> *da, INTEGER ldda,
                             std::complex<double> *db, INTEGER lddb, INTEGER *info);





#endif // USE_CXXMAGMA

#ifdef __cplusplus
} // extern C
#endif

#endif // PLAYGROUND_CXXMAGMA_HEADER_H 1

