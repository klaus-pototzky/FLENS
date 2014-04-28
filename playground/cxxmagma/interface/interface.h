/*
 *   Copyright (c) 2012, Michael Lehn
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_INTERFACE_H
#define PLAYGROUND_CXXMAGMA_INTERFACE_INTERFACE_H 1

#include <complex>

// #include <playground/cxxmagma/interface/bbcsd.h>
// #include <playground/cxxmagma/interface/bdsdc.h>
// #include <playground/cxxmagma/interface/bdsqr.h>
// #include <playground/cxxmagma/interface/cgesv.h>
// #include <playground/cxxmagma/interface/chla_transtype.h>
// #include <playground/cxxmagma/interface/cposv.h>
// #include <playground/cxxmagma/interface/disna.h>
// #include <playground/cxxmagma/interface/dspcon.h>
// #include <playground/cxxmagma/interface/gbbrd.h>
// #include <playground/cxxmagma/interface/gbcon.h>
// #include <playground/cxxmagma/interface/gbequb.h>
// #include <playground/cxxmagma/interface/gbequ.h>
// #include <playground/cxxmagma/interface/gbrfs.h>
// #include <playground/cxxmagma/interface/gbsv.h>
// #include <playground/cxxmagma/interface/gbsvx.h>
// #include <playground/cxxmagma/interface/gbtf2.h>
// #include <playground/cxxmagma/interface/gbtrf.h>
// #include <playground/cxxmagma/interface/gbtrs.h>
// #include <playground/cxxmagma/interface/gebak.h>
// #include <playground/cxxmagma/interface/gebal.h>
// #include <playground/cxxmagma/interface/gebd2.h>
// #include <playground/cxxmagma/interface/gebrd.h>
// #include <playground/cxxmagma/interface/gecon.h>
// #include <playground/cxxmagma/interface/geequb.h>
// #include <playground/cxxmagma/interface/geequ.h>
// #include <playground/cxxmagma/interface/gees.h>
// #include <playground/cxxmagma/interface/geesx.h>
#include <playground/cxxmagma/interface/geev.h>
// #include <playground/cxxmagma/interface/geevx.h>
// #include <playground/cxxmagma/interface/gegs.h>
// #include <playground/cxxmagma/interface/gegv.h>
// #include <playground/cxxmagma/interface/gehd2.h>
#include <playground/cxxmagma/interface/gehrd.h>
// #include <playground/cxxmagma/interface/gejsv.h>
// #include <playground/cxxmagma/interface/gelq2.h>
#include <playground/cxxmagma/interface/gelqf.h>
// #include <playground/cxxmagma/interface/gelsd.h>
// #include <playground/cxxmagma/interface/gels.h>
// #include <playground/cxxmagma/interface/gelss.h>
// #include <playground/cxxmagma/interface/gelsx.h>
// #include <playground/cxxmagma/interface/gelsy.h>
// #include <playground/cxxmagma/interface/geql2.h>
#include <playground/cxxmagma/interface/geqlf.h>
// #include <playground/cxxmagma/interface/geqp3.h>
// #include <playground/cxxmagma/interface/geqpf.h>
// #include <playground/cxxmagma/interface/geqr2.h>
// #include <playground/cxxmagma/interface/geqr2p.h>
#include <playground/cxxmagma/interface/geqrf.h>
// #include <playground/cxxmagma/interface/geqrfp.h>
// #include <playground/cxxmagma/interface/gerfs.h>
// #include <playground/cxxmagma/interface/gerq2.h>
// #include <playground/cxxmagma/interface/gerqf.h>
// #include <playground/cxxmagma/interface/gesc2.h>
// #include <playground/cxxmagma/interface/gesdd.h>
#include <playground/cxxmagma/interface/gesvd.h>
#include <playground/cxxmagma/interface/gesv.h>
// #include <playground/cxxmagma/interface/gesvj.h>
// #include <playground/cxxmagma/interface/gesvx.h>
// #include <playground/cxxmagma/interface/getc2.h>
// #include <playground/cxxmagma/interface/getf2.h>
#include <playground/cxxmagma/interface/getrf.h>
#include <playground/cxxmagma/interface/getri.h>
#include <playground/cxxmagma/interface/getrs.h>
// #include <playground/cxxmagma/interface/ggbak.h>
// #include <playground/cxxmagma/interface/ggbal.h>
// #include <playground/cxxmagma/interface/gges.h>
// #include <playground/cxxmagma/interface/ggesx.h>
// #include <playground/cxxmagma/interface/ggev.h>
// #include <playground/cxxmagma/interface/ggevx.h>
// #include <playground/cxxmagma/interface/ggglm.h>
// #include <playground/cxxmagma/interface/gghrd.h>
// #include <playground/cxxmagma/interface/gglse.h>
// #include <playground/cxxmagma/interface/ggqrf.h>
// #include <playground/cxxmagma/interface/ggrqf.h>
// #include <playground/cxxmagma/interface/ggsvd.h>
// #include <playground/cxxmagma/interface/ggsvp.h>
// #include <playground/cxxmagma/interface/gsvj0.h>
// #include <playground/cxxmagma/interface/gsvj1.h>
// #include <playground/cxxmagma/interface/gtcon.h>
// #include <playground/cxxmagma/interface/gtrfs.h>
// #include <playground/cxxmagma/interface/gtsv.h>
// #include <playground/cxxmagma/interface/gtsvx.h>
// #include <playground/cxxmagma/interface/gttrf.h>
// #include <playground/cxxmagma/interface/gttrs.h>
// #include <playground/cxxmagma/interface/gtts2.h>
// #include <playground/cxxmagma/interface/hbevd.h>
// #include <playground/cxxmagma/interface/hbev.h>
// #include <playground/cxxmagma/interface/hbevx.h>
// #include <playground/cxxmagma/interface/hbgst.h>
// #include <playground/cxxmagma/interface/hbgvd.h>
// #include <playground/cxxmagma/interface/hbgv.h>
// #include <playground/cxxmagma/interface/hbgvx.h>
// #include <playground/cxxmagma/interface/hbtrd.h>
// #include <playground/cxxmagma/interface/hecon.h>
// #include <playground/cxxmagma/interface/heequb.h>
// #include <playground/cxxmagma/interface/heevd.h>
// #include <playground/cxxmagma/interface/heev.h>
// #include <playground/cxxmagma/interface/heevr.h>
// #include <playground/cxxmagma/interface/heevx.h>
// #include <playground/cxxmagma/interface/hegs2.h>
// #include <playground/cxxmagma/interface/hegst.h>
// #include <playground/cxxmagma/interface/hegvd.h>
// #include <playground/cxxmagma/interface/hegv.h>
// #include <playground/cxxmagma/interface/hegvx.h>
// #include <playground/cxxmagma/interface/herfs.h>
// #include <playground/cxxmagma/interface/hesv.h>
// #include <playground/cxxmagma/interface/hesvx.h>
// #include <playground/cxxmagma/interface/heswapr.h>
// #include <playground/cxxmagma/interface/hetd2.h>
// #include <playground/cxxmagma/interface/hetf2.h>
// #include <playground/cxxmagma/interface/hetrd.h>
// #include <playground/cxxmagma/interface/hetrf.h>
// #include <playground/cxxmagma/interface/hetri2.h>
// #include <playground/cxxmagma/interface/hetri2x.h>
// #include <playground/cxxmagma/interface/hetri.h>
// #include <playground/cxxmagma/interface/hetrs2.h>
// #include <playground/cxxmagma/interface/hetrs.h>
// #include <playground/cxxmagma/interface/hfrk.h>
// #include <playground/cxxmagma/interface/hgeqz.h>
// #include <playground/cxxmagma/interface/hpcon.h>
// #include <playground/cxxmagma/interface/hpevd.h>
// #include <playground/cxxmagma/interface/hpev.h>
// #include <playground/cxxmagma/interface/hpevx.h>
// #include <playground/cxxmagma/interface/hpgst.h>
// #include <playground/cxxmagma/interface/hpgvd.h>
// #include <playground/cxxmagma/interface/hpgv.h>
// #include <playground/cxxmagma/interface/hpgvx.h>
// #include <playground/cxxmagma/interface/hprfs.h>
// #include <playground/cxxmagma/interface/hpsv.h>
// #include <playground/cxxmagma/interface/hpsvx.h>
// #include <playground/cxxmagma/interface/hptrd.h>
// #include <playground/cxxmagma/interface/hptrf.h>
// #include <playground/cxxmagma/interface/hptri.h>
// #include <playground/cxxmagma/interface/hptrs.h>
// #include <playground/cxxmagma/interface/hsein.h>
// #include <playground/cxxmagma/interface/hseqr.h>
// #include <playground/cxxmagma/interface/ieeeck.h>
// #include <playground/cxxmagma/interface/iladlc.h>
// #include <playground/cxxmagma/interface/iladlr.h>
// #include <playground/cxxmagma/interface/ilalc.h>
// #include <playground/cxxmagma/interface/ilalr.h>
// #include <playground/cxxmagma/interface/laprec.h>
// #include <playground/cxxmagma/interface/ilaslc.h>
// #include <playground/cxxmagma/interface/ilaslr.h>
// #include <playground/cxxmagma/interface/latrans.h>
// #include <playground/cxxmagma/interface/lauplo.h>
// #include <playground/cxxmagma/interface/ilaver.h>
// #include <playground/cxxmagma/interface/ilazlc.h>
// #include <playground/cxxmagma/interface/ilazlr.h>
// #include <playground/cxxmagma/interface/interface.h>
// #include <playground/cxxmagma/interface/isnan.h>
// #include <playground/cxxmagma/interface/izmax1.h>
// #include <playground/cxxmagma/interface/labad.h>
// #include <playground/cxxmagma/interface/labrd.h>
// #include <playground/cxxmagma/interface/lacgv.h>
// #include <playground/cxxmagma/interface/lacn2.h>
// #include <playground/cxxmagma/interface/lacon.h>
// #include <playground/cxxmagma/interface/lacp2.h>
// #include <playground/cxxmagma/interface/lacpy.h>
// #include <playground/cxxmagma/interface/lacrm.h>
// #include <playground/cxxmagma/interface/lacrt.h>
// #include <playground/cxxmagma/interface/ladiv.h>
// #include <playground/cxxmagma/interface/lae2.h>
// #include <playground/cxxmagma/interface/laebz.h>
// #include <playground/cxxmagma/interface/laed0.h>
// #include <playground/cxxmagma/interface/laed1.h>
// #include <playground/cxxmagma/interface/laed2.h>
// #include <playground/cxxmagma/interface/laed3.h>
// #include <playground/cxxmagma/interface/laed4.h>
// #include <playground/cxxmagma/interface/laed5.h>
// #include <playground/cxxmagma/interface/laed6.h>
// #include <playground/cxxmagma/interface/laed7.h>
// #include <playground/cxxmagma/interface/laed8.h>
// #include <playground/cxxmagma/interface/laed9.h>
// #include <playground/cxxmagma/interface/laeda.h>
// #include <playground/cxxmagma/interface/laein.h>
// #include <playground/cxxmagma/interface/laesy.h>
// #include <playground/cxxmagma/interface/laev2.h>
// #include <playground/cxxmagma/interface/laexc.h>
// #include <playground/cxxmagma/interface/lag2c.h>
// #include <playground/cxxmagma/interface/lag2d.h>
// #include <playground/cxxmagma/interface/lag2.h>
// #include <playground/cxxmagma/interface/lag2s.h>
// #include <playground/cxxmagma/interface/lag2z.h>
// #include <playground/cxxmagma/interface/la_gbamv.h>
// #include <playground/cxxmagma/interface/la_gbrcond_c.h>
// #include <playground/cxxmagma/interface/la_gbrcond.h>
// #include <playground/cxxmagma/interface/la_gbrcond_x.h>
// #include <playground/cxxmagma/interface/la_gbrpvgrw.h>
// #include <playground/cxxmagma/interface/la_geamv.h>
// #include <playground/cxxmagma/interface/la_gercond_c.h>
// #include <playground/cxxmagma/interface/la_gercond.h>
// #include <playground/cxxmagma/interface/la_gercond_x.h>
// #include <playground/cxxmagma/interface/lags2.h>
// #include <playground/cxxmagma/interface/lagtf.h>
// #include <playground/cxxmagma/interface/lagtm.h>
// #include <playground/cxxmagma/interface/lagts.h>
// #include <playground/cxxmagma/interface/lagv2.h>
// #include <playground/cxxmagma/interface/lahef.h>
// #include <playground/cxxmagma/interface/la_heramv.h>
// #include <playground/cxxmagma/interface/la_hercond_c.h>
// #include <playground/cxxmagma/interface/la_hercond_x.h>
// #include <playground/cxxmagma/interface/la_herpvgrw.h>
// #include <playground/cxxmagma/interface/lahqr.h>
// #include <playground/cxxmagma/interface/lahr2.h>
// #include <playground/cxxmagma/interface/lahrd.h>
// #include <playground/cxxmagma/interface/laic1.h>
// #include <playground/cxxmagma/interface/laisnan.h>
// #include <playground/cxxmagma/interface/la_lin_berr.h>
// #include <playground/cxxmagma/interface/laln2.h>
// #include <playground/cxxmagma/interface/lals0.h>
// #include <playground/cxxmagma/interface/lalsa.h>
// #include <playground/cxxmagma/interface/lalsd.h>
// #include <playground/cxxmagma/interface/lamch.h>
// #include <playground/cxxmagma/interface/lamrg.h>
// #include <playground/cxxmagma/interface/laneg.h>
// #include <playground/cxxmagma/interface/langb.h>
// #include <playground/cxxmagma/interface/lange.h>
// #include <playground/cxxmagma/interface/langt.h>
// #include <playground/cxxmagma/interface/lanhb.h>
// #include <playground/cxxmagma/interface/lanhe.h>
// #include <playground/cxxmagma/interface/lanhf.h>
// #include <playground/cxxmagma/interface/lanhp.h>
// #include <playground/cxxmagma/interface/lanhs.h>
// #include <playground/cxxmagma/interface/lanht.h>
// #include <playground/cxxmagma/interface/lansb.h>
// #include <playground/cxxmagma/interface/lansf.h>
// #include <playground/cxxmagma/interface/lansp.h>
// #include <playground/cxxmagma/interface/lanst.h>
// #include <playground/cxxmagma/interface/lansy.h>
// #include <playground/cxxmagma/interface/lantb.h>
// #include <playground/cxxmagma/interface/lantp.h>
// #include <playground/cxxmagma/interface/lantr.h>
// #include <playground/cxxmagma/interface/lanv2.h>
// #include <playground/cxxmagma/interface/lapll.h>
// #include <playground/cxxmagma/interface/lapmr.h>
// #include <playground/cxxmagma/interface/lapmt.h>
// #include <playground/cxxmagma/interface/la_porcond_c.h>
// #include <playground/cxxmagma/interface/la_porcond.h>
// #include <playground/cxxmagma/interface/la_porcond_x.h>
// #include <playground/cxxmagma/interface/la_porpvgrw.h>
// #include <playground/cxxmagma/interface/lapy2.h>
// #include <playground/cxxmagma/interface/lapy3.h>
// #include <playground/cxxmagma/interface/laqgb.h>
// #include <playground/cxxmagma/interface/laqge.h>
// #include <playground/cxxmagma/interface/laqhb.h>
// #include <playground/cxxmagma/interface/laqhe.h>
// #include <playground/cxxmagma/interface/laqhp.h>
// #include <playground/cxxmagma/interface/laqp2.h>
// #include <playground/cxxmagma/interface/laqps.h>
// #include <playground/cxxmagma/interface/laqr0.h>
// #include <playground/cxxmagma/interface/laqr1.h>
// #include <playground/cxxmagma/interface/laqr2.h>
// #include <playground/cxxmagma/interface/laqr3.h>
// #include <playground/cxxmagma/interface/laqr4.h>
// #include <playground/cxxmagma/interface/laqr5.h>
// #include <playground/cxxmagma/interface/laqsb.h>
// #include <playground/cxxmagma/interface/laqsp.h>
// #include <playground/cxxmagma/interface/laqsy.h>
// #include <playground/cxxmagma/interface/laqtr.h>
// #include <playground/cxxmagma/interface/lar1v.h>
// #include <playground/cxxmagma/interface/lar2v.h>
// #include <playground/cxxmagma/interface/larcm.h>
// #include <playground/cxxmagma/interface/larfb.h>
// #include <playground/cxxmagma/interface/larfg.h>
// #include <playground/cxxmagma/interface/larfgp.h>
// #include <playground/cxxmagma/interface/larf.h>
// #include <playground/cxxmagma/interface/larft.h>
// #include <playground/cxxmagma/interface/larfx.h>
// #include <playground/cxxmagma/interface/largv.h>
// #include <playground/cxxmagma/interface/larnv.h>
// #include <playground/cxxmagma/interface/la_rpvgrw.h>
// #include <playground/cxxmagma/interface/larra.h>
// #include <playground/cxxmagma/interface/larrb.h>
// #include <playground/cxxmagma/interface/larrc.h>
// #include <playground/cxxmagma/interface/larrd.h>
// #include <playground/cxxmagma/interface/larre.h>
// #include <playground/cxxmagma/interface/larrf.h>
// #include <playground/cxxmagma/interface/larrj.h>
// #include <playground/cxxmagma/interface/larrk.h>
// #include <playground/cxxmagma/interface/larrr.h>
// #include <playground/cxxmagma/interface/larrv.h>
// #include <playground/cxxmagma/interface/larscl2.h>
// #include <playground/cxxmagma/interface/lartg.h>
// #include <playground/cxxmagma/interface/lartgp.h>
// #include <playground/cxxmagma/interface/lartgs.h>
// #include <playground/cxxmagma/interface/lartv.h>
// #include <playground/cxxmagma/interface/laruv.h>
// #include <playground/cxxmagma/interface/larzb.h>
// #include <playground/cxxmagma/interface/larz.h>
// #include <playground/cxxmagma/interface/larzt.h>
// #include <playground/cxxmagma/interface/las2.h>
// #include <playground/cxxmagma/interface/lascl2.h>
// #include <playground/cxxmagma/interface/lascl.h>
// #include <playground/cxxmagma/interface/lasd0.h>
// #include <playground/cxxmagma/interface/lasd1.h>
// #include <playground/cxxmagma/interface/lasd2.h>
// #include <playground/cxxmagma/interface/lasd3.h>
// #include <playground/cxxmagma/interface/lasd4.h>
// #include <playground/cxxmagma/interface/lasd5.h>
// #include <playground/cxxmagma/interface/lasd6.h>
// #include <playground/cxxmagma/interface/lasd7.h>
// #include <playground/cxxmagma/interface/lasd8.h>
// #include <playground/cxxmagma/interface/lasda.h>
// #include <playground/cxxmagma/interface/lasdq.h>
// #include <playground/cxxmagma/interface/lasdt.h>
// #include <playground/cxxmagma/interface/laset.h>
// #include <playground/cxxmagma/interface/lasq1.h>
// #include <playground/cxxmagma/interface/lasq2.h>
// #include <playground/cxxmagma/interface/lasq3.h>
// #include <playground/cxxmagma/interface/lasq4.h>
// #include <playground/cxxmagma/interface/lasq5.h>
// #include <playground/cxxmagma/interface/lasq6.h>
// #include <playground/cxxmagma/interface/lasr.h>
// #include <playground/cxxmagma/interface/lasrt.h>
// #include <playground/cxxmagma/interface/lassq.h>
// #include <playground/cxxmagma/interface/lasv2.h>
// #include <playground/cxxmagma/interface/laswp.h>
// #include <playground/cxxmagma/interface/lasy2.h>
// #include <playground/cxxmagma/interface/la_syamv.h>
// #include <playground/cxxmagma/interface/lasyf.h>
// #include <playground/cxxmagma/interface/la_syrcond_c.h>
// #include <playground/cxxmagma/interface/la_syrcond.h>
// #include <playground/cxxmagma/interface/la_syrcond_x.h>
// #include <playground/cxxmagma/interface/la_syrpvgrw.h>
// #include <playground/cxxmagma/interface/lat2c.h>
// #include <playground/cxxmagma/interface/lat2s.h>
// #include <playground/cxxmagma/interface/latbs.h>
// #include <playground/cxxmagma/interface/latdf.h>
// #include <playground/cxxmagma/interface/latps.h>
// #include <playground/cxxmagma/interface/latrd.h>
// #include <playground/cxxmagma/interface/latrs.h>
// #include <playground/cxxmagma/interface/latrz.h>
// #include <playground/cxxmagma/interface/latzm.h>
// #include <playground/cxxmagma/interface/lauu2.h>
// #include <playground/cxxmagma/interface/lauum.h>
// #include <playground/cxxmagma/interface/la_wwaddw.h>
// #include <playground/cxxmagma/interface/lsame.h>
// #include <playground/cxxmagma/interface/lsamen.h>
// #include <playground/cxxmagma/interface/opgtr.h>
// #include <playground/cxxmagma/interface/opmtr.h>
// #include <playground/cxxmagma/interface/orbdb.h>
// #include <playground/cxxmagma/interface/orcsd.h>
// #include <playground/cxxmagma/interface/org2l.h>
// #include <playground/cxxmagma/interface/org2r.h>
// #include <playground/cxxmagma/interface/orgbr.h>
// #include <playground/cxxmagma/interface/orghr.h>
// #include <playground/cxxmagma/interface/orgl2.h>
// #include <playground/cxxmagma/interface/orglq.h>
// #include <playground/cxxmagma/interface/orgql.h>
// #include <playground/cxxmagma/interface/orgqr.h>
// #include <playground/cxxmagma/interface/orgr2.h>
// #include <playground/cxxmagma/interface/orgrq.h>
// #include <playground/cxxmagma/interface/orgtr.h>
// #include <playground/cxxmagma/interface/orm2l.h>
// #include <playground/cxxmagma/interface/orm2r.h>
// #include <playground/cxxmagma/interface/ormbr.h>
// #include <playground/cxxmagma/interface/ormhr.h>
// #include <playground/cxxmagma/interface/orml2.h>
// #include <playground/cxxmagma/interface/ormlq.h>
#include <playground/cxxmagma/interface/ormql.h>
#include <playground/cxxmagma/interface/ormqr.h>
// #include <playground/cxxmagma/interface/ormr2.h>
// #include <playground/cxxmagma/interface/ormr3.h>
// #include <playground/cxxmagma/interface/ormrq.h>
// #include <playground/cxxmagma/interface/ormrz.h>
// #include <playground/cxxmagma/interface/ormtr.h>
// #include <playground/cxxmagma/interface/pbcon.h>
// #include <playground/cxxmagma/interface/pbequ.h>
// #include <playground/cxxmagma/interface/pbrfs.h>
// #include <playground/cxxmagma/interface/pbstf.h>
// #include <playground/cxxmagma/interface/pbsv.h>
// #include <playground/cxxmagma/interface/pbsvx.h>
// #include <playground/cxxmagma/interface/pbtf2.h>
// #include <playground/cxxmagma/interface/pbtrf.h>
// #include <playground/cxxmagma/interface/pbtrs.h>
// #include <playground/cxxmagma/interface/pftrf.h>
// #include <playground/cxxmagma/interface/pftri.h>
// #include <playground/cxxmagma/interface/pftrs.h>
// #include <playground/cxxmagma/interface/pocon.h>
// #include <playground/cxxmagma/interface/poequb.h>
// #include <playground/cxxmagma/interface/poequ.h>
// #include <playground/cxxmagma/interface/porfs.h>
#include <playground/cxxmagma/interface/posv.h>
// #include <playground/cxxmagma/interface/posvx.h>
// #include <playground/cxxmagma/interface/potf2.h>
#include <playground/cxxmagma/interface/potrf.h>
#include <playground/cxxmagma/interface/potri.h>
// #include <playground/cxxmagma/interface/potrs.h>
// #include <playground/cxxmagma/interface/ppcon.h>
// #include <playground/cxxmagma/interface/ppequ.h>
// #include <playground/cxxmagma/interface/pprfs.h>
// #include <playground/cxxmagma/interface/ppsv.h>
// #include <playground/cxxmagma/interface/ppsvx.h>
// #include <playground/cxxmagma/interface/pptrf.h>
// #include <playground/cxxmagma/interface/pptri.h>
// #include <playground/cxxmagma/interface/pptrs.h>
// #include <playground/cxxmagma/interface/pstf2.h>
// #include <playground/cxxmagma/interface/pstrf.h>
// #include <playground/cxxmagma/interface/ptcon.h>
// #include <playground/cxxmagma/interface/pteqr.h>
// #include <playground/cxxmagma/interface/ptrfs.h>
// #include <playground/cxxmagma/interface/ptsv.h>
// #include <playground/cxxmagma/interface/ptsvx.h>
// #include <playground/cxxmagma/interface/pttrf.h>
// #include <playground/cxxmagma/interface/pttrs.h>
// #include <playground/cxxmagma/interface/ptts2.h>
// #include <playground/cxxmagma/interface/rot.h>
// #include <playground/cxxmagma/interface/rscl.h>
// #include <playground/cxxmagma/interface/sbevd.h>
// #include <playground/cxxmagma/interface/sbev.h>
// #include <playground/cxxmagma/interface/sbevx.h>
// #include <playground/cxxmagma/interface/sbgst.h>
// #include <playground/cxxmagma/interface/sbgvd.h>
// #include <playground/cxxmagma/interface/sbgv.h>
// #include <playground/cxxmagma/interface/sbgvx.h>
// #include <playground/cxxmagma/interface/sbtrd.h>
// #include <playground/cxxmagma/interface/sfrk.h>
// #include <playground/cxxmagma/interface/sgesv.h>
// #include <playground/cxxmagma/interface/spevd.h>
// #include <playground/cxxmagma/interface/spev.h>
// #include <playground/cxxmagma/interface/spevx.h>
// #include <playground/cxxmagma/interface/spgst.h>
// #include <playground/cxxmagma/interface/spgvd.h>
// #include <playground/cxxmagma/interface/spgv.h>
// #include <playground/cxxmagma/interface/spgvx.h>
// #include <playground/cxxmagma/interface/spmv.h>
// #include <playground/cxxmagma/interface/sposv.h>
// #include <playground/cxxmagma/interface/sprfs.h>
// #include <playground/cxxmagma/interface/spr.h>
// #include <playground/cxxmagma/interface/spsv.h>
// #include <playground/cxxmagma/interface/spsvx.h>
// #include <playground/cxxmagma/interface/sptrd.h>
// #include <playground/cxxmagma/interface/sptrf.h>
// #include <playground/cxxmagma/interface/sptri.h>
// #include <playground/cxxmagma/interface/sptrs.h>
// #include <playground/cxxmagma/interface/stebz.h>
// #include <playground/cxxmagma/interface/stedc.h>
// #include <playground/cxxmagma/interface/stegr.h>
// #include <playground/cxxmagma/interface/stein.h>
// #include <playground/cxxmagma/interface/stemr.h>
// #include <playground/cxxmagma/interface/steqr.h>
// #include <playground/cxxmagma/interface/sterf.h>
// #include <playground/cxxmagma/interface/stevd.h>
// #include <playground/cxxmagma/interface/stev.h>
// #include <playground/cxxmagma/interface/stevr.h>
// #include <playground/cxxmagma/interface/stevx.h>
// #include <playground/cxxmagma/interface/sycon.h>
// #include <playground/cxxmagma/interface/syconv.h>
// #include <playground/cxxmagma/interface/syequb.h>
// #include <playground/cxxmagma/interface/syevd.h>
// #include <playground/cxxmagma/interface/syev.h>
// #include <playground/cxxmagma/interface/syevr.h>
// #include <playground/cxxmagma/interface/syevx.h>
// #include <playground/cxxmagma/interface/sygs2.h>
// #include <playground/cxxmagma/interface/sygst.h>
// #include <playground/cxxmagma/interface/sygvd.h>
// #include <playground/cxxmagma/interface/sygv.h>
// #include <playground/cxxmagma/interface/sygvx.h>
// #include <playground/cxxmagma/interface/symv.h>
// #include <playground/cxxmagma/interface/syrfs.h>
// #include <playground/cxxmagma/interface/syr.h>
// #include <playground/cxxmagma/interface/sysv.h>
// #include <playground/cxxmagma/interface/sysvx.h>
// #include <playground/cxxmagma/interface/syswapr.h>
// #include <playground/cxxmagma/interface/sytd2.h>
// #include <playground/cxxmagma/interface/sytf2.h>
// #include <playground/cxxmagma/interface/sytrd.h>
// #include <playground/cxxmagma/interface/sytrf.h>
// #include <playground/cxxmagma/interface/sytri2.h>
// #include <playground/cxxmagma/interface/sytri2x.h>
// #include <playground/cxxmagma/interface/sytri.h>
// #include <playground/cxxmagma/interface/sytrs2.h>
// #include <playground/cxxmagma/interface/sytrs.h>
// #include <playground/cxxmagma/interface/tbcon.h>
// #include <playground/cxxmagma/interface/tbrfs.h>
// #include <playground/cxxmagma/interface/tbtrs.h>
// #include <playground/cxxmagma/interface/tfsm.h>
// #include <playground/cxxmagma/interface/tftri.h>
// #include <playground/cxxmagma/interface/tfttp.h>
// #include <playground/cxxmagma/interface/tfttr.h>
// #include <playground/cxxmagma/interface/tgevc.h>
// #include <playground/cxxmagma/interface/tgex2.h>
// #include <playground/cxxmagma/interface/tgexc.h>
// #include <playground/cxxmagma/interface/tgsen.h>
// #include <playground/cxxmagma/interface/tgsja.h>
// #include <playground/cxxmagma/interface/tgsna.h>
// #include <playground/cxxmagma/interface/tgsy2.h>
// #include <playground/cxxmagma/interface/tgsyl.h>
// #include <playground/cxxmagma/interface/tpcon.h>
// #include <playground/cxxmagma/interface/tprfs.h>
// #include <playground/cxxmagma/interface/tptri.h>
// #include <playground/cxxmagma/interface/tptrs.h>
// #include <playground/cxxmagma/interface/tpttf.h>
// #include <playground/cxxmagma/interface/tpttr.h>
// #include <playground/cxxmagma/interface/trcon.h>
// #include <playground/cxxmagma/interface/trevc.h>
// #include <playground/cxxmagma/interface/trexc.h>
// #include <playground/cxxmagma/interface/trrfs.h>
// #include <playground/cxxmagma/interface/trsen.h>
// #include <playground/cxxmagma/interface/trsna.h>
// #include <playground/cxxmagma/interface/trsyl.h>
// #include <playground/cxxmagma/interface/trti2.h>
// #include <playground/cxxmagma/interface/trtri.h>
// #include <playground/cxxmagma/interface/trtrs.h>
// #include <playground/cxxmagma/interface/trttf.h>
// #include <playground/cxxmagma/interface/trttp.h>
// #include <playground/cxxmagma/interface/tzrqf.h>
// #include <playground/cxxmagma/interface/tzrzf.h>
// #include <playground/cxxmagma/interface/unbdb.h>
// #include <playground/cxxmagma/interface/uncsd.h>
// #include <playground/cxxmagma/interface/ung2l.h>
// #include <playground/cxxmagma/interface/ung2r.h>
// #include <playground/cxxmagma/interface/ungbr.h>
// #include <playground/cxxmagma/interface/unghr.h>
// #include <playground/cxxmagma/interface/ungl2.h>
// #include <playground/cxxmagma/interface/unglq.h>
// #include <playground/cxxmagma/interface/ungql.h>
// #include <playground/cxxmagma/interface/ungqr.h>
// #include <playground/cxxmagma/interface/ungr2.h>
// #include <playground/cxxmagma/interface/ungrq.h>
// #include <playground/cxxmagma/interface/ungtr.h>
// #include <playground/cxxmagma/interface/unm2l.h>
// #include <playground/cxxmagma/interface/unm2r.h>
// #include <playground/cxxmagma/interface/unmbr.h>
// #include <playground/cxxmagma/interface/unmhr.h>
// #include <playground/cxxmagma/interface/unml2.h>
// #include <playground/cxxmagma/interface/unmlq.h>
#include <playground/cxxmagma/interface/unmql.h>
#include <playground/cxxmagma/interface/unmqr.h>
// #include <playground/cxxmagma/interface/unmr2.h>
// #include <playground/cxxmagma/interface/unmr3.h>
// #include <playground/cxxmagma/interface/unmrq.h>
// #include <playground/cxxmagma/interface/unmrz.h>
// #include <playground/cxxmagma/interface/unmtr.h>
// #include <playground/cxxmagma/interface/upgtr.h>
// #include <playground/cxxmagma/interface/upmtr.h>
// #include <playground/cxxmagma/interface/zsum1.h>

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_INTERFACE_H
