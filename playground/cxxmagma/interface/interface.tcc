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

#ifndef PLAYGROUND_CXXMAGMA_INTERFACE_INTERFACE_TCC
#define PLAYGROUND_CXXMAGMA_INTERFACE_INTERFACE_TCC 1

// #include <playground/cxxmagma/interface/bbcsd.tcc>
// #include <playground/cxxmagma/interface/bdsdc.tcc>
// #include <playground/cxxmagma/interface/bdsqr.tcc>
// #include <playground/cxxmagma/interface/cgesv.tcc>
// #include <playground/cxxmagma/interface/chla_transtype.tcc>
// #include <playground/cxxmagma/interface/cposv.tcc>
// #include <playground/cxxmagma/interface/disna.tcc>
// #include <playground/cxxmagma/interface/dspcon.tcc>
// #include <playground/cxxmagma/interface/gbbrd.tcc>
// #include <playground/cxxmagma/interface/gbcon.tcc>
// #include <playground/cxxmagma/interface/gbequb.tcc>
// #include <playground/cxxmagma/interface/gbequ.tcc>
// #include <playground/cxxmagma/interface/gbrfs.tcc>
// #include <playground/cxxmagma/interface/gbsv.tcc>
// #include <playground/cxxmagma/interface/gbsvx.tcc>
// #include <playground/cxxmagma/interface/gbtf2.tcc>
// #include <playground/cxxmagma/interface/gbtrf.tcc>
// #include <playground/cxxmagma/interface/gbtrs.tcc>
// #include <playground/cxxmagma/interface/gebak.tcc>
// #include <playground/cxxmagma/interface/gebal.tcc>
// #include <playground/cxxmagma/interface/gebd2.tcc>
// #include <playground/cxxmagma/interface/gebrd.tcc>
// #include <playground/cxxmagma/interface/gecon.tcc>
// #include <playground/cxxmagma/interface/geequb.tcc>
// #include <playground/cxxmagma/interface/geequ.tcc>
// #include <playground/cxxmagma/interface/gees.tcc>
// #include <playground/cxxmagma/interface/geesx.tcc>
#include <playground/cxxmagma/interface/geev.tcc>
// #include <playground/cxxmagma/interface/geevx.tcc>
// #include <playground/cxxmagma/interface/gegs.tcc>
// #include <playground/cxxmagma/interface/gegv.tcc>
// #include <playground/cxxmagma/interface/gehd2.tcc>
#include <playground/cxxmagma/interface/gehrd.tcc>
// #include <playground/cxxmagma/interface/gejsv.tcc>
// #include <playground/cxxmagma/interface/gelq2.tcc>
#include <playground/cxxmagma/interface/gelqf.tcc>
// #include <playground/cxxmagma/interface/gelsd.tcc>
// #include <playground/cxxmagma/interface/gels.tcc>
// #include <playground/cxxmagma/interface/gelss.tcc>
// #include <playground/cxxmagma/interface/gelsx.tcc>
// #include <playground/cxxmagma/interface/gelsy.tcc>
// #include <playground/cxxmagma/interface/geql2.tcc>
#include <playground/cxxmagma/interface/geqlf.tcc>
// #include <playground/cxxmagma/interface/geqp3.tcc>
// #include <playground/cxxmagma/interface/geqpf.tcc>
// #include <playground/cxxmagma/interface/geqr2.tcc>
// #include <playground/cxxmagma/interface/geqr2p.tcc>
#include <playground/cxxmagma/interface/geqrf.tcc>
// #include <playground/cxxmagma/interface/geqrfp.tcc>
// #include <playground/cxxmagma/interface/gerfs.tcc>
// #include <playground/cxxmagma/interface/gerq2.tcc>
// #include <playground/cxxmagma/interface/gerqf.tcc>
// #include <playground/cxxmagma/interface/gesc2.tcc>
// #include <playground/cxxmagma/interface/gesdd.tcc>
#include <playground/cxxmagma/interface/gesvd.tcc>
#include <playground/cxxmagma/interface/gesv.tcc>
// #include <playground/cxxmagma/interface/gesvj.tcc>
// #include <playground/cxxmagma/interface/gesvx.tcc>
// #include <playground/cxxmagma/interface/getc2.tcc>
// #include <playground/cxxmagma/interface/getf2.tcc>
#include <playground/cxxmagma/interface/getrf.tcc>
#include <playground/cxxmagma/interface/getri.tcc>
#include <playground/cxxmagma/interface/getrs.tcc>
// #include <playground/cxxmagma/interface/ggbak.tcc>
// #include <playground/cxxmagma/interface/ggbal.tcc>
// #include <playground/cxxmagma/interface/gges.tcc>
// #include <playground/cxxmagma/interface/ggesx.tcc>
// #include <playground/cxxmagma/interface/ggev.tcc>
// #include <playground/cxxmagma/interface/ggevx.tcc>
// #include <playground/cxxmagma/interface/ggglm.tcc>
// #include <playground/cxxmagma/interface/gghrd.tcc>
// #include <playground/cxxmagma/interface/gglse.tcc>
// #include <playground/cxxmagma/interface/ggqrf.tcc>
// #include <playground/cxxmagma/interface/ggrqf.tcc>
// #include <playground/cxxmagma/interface/ggsvd.tcc>
// #include <playground/cxxmagma/interface/ggsvp.tcc>
// #include <playground/cxxmagma/interface/gsvj0.tcc>
// #include <playground/cxxmagma/interface/gsvj1.tcc>
// #include <playground/cxxmagma/interface/gtcon.tcc>
// #include <playground/cxxmagma/interface/gtrfs.tcc>
// #include <playground/cxxmagma/interface/gtsv.tcc>
// #include <playground/cxxmagma/interface/gtsvx.tcc>
// #include <playground/cxxmagma/interface/gttrf.tcc>
// #include <playground/cxxmagma/interface/gttrs.tcc>
// #include <playground/cxxmagma/interface/gtts2.tcc>
// #include <playground/cxxmagma/interface/hbevd.tcc>
// #include <playground/cxxmagma/interface/hbev.tcc>
// #include <playground/cxxmagma/interface/hbevx.tcc>
// #include <playground/cxxmagma/interface/hbgst.tcc>
// #include <playground/cxxmagma/interface/hbgvd.tcc>
// #include <playground/cxxmagma/interface/hbgv.tcc>
// #include <playground/cxxmagma/interface/hbgvx.tcc>
// #include <playground/cxxmagma/interface/hbtrd.tcc>
// #include <playground/cxxmagma/interface/hecon.tcc>
// #include <playground/cxxmagma/interface/heequb.tcc>
// #include <playground/cxxmagma/interface/heevd.tcc>
// #include <playground/cxxmagma/interface/heev.tcc>
// #include <playground/cxxmagma/interface/heevr.tcc>
// #include <playground/cxxmagma/interface/heevx.tcc>
// #include <playground/cxxmagma/interface/hegs2.tcc>
// #include <playground/cxxmagma/interface/hegst.tcc>
// #include <playground/cxxmagma/interface/hegvd.tcc>
// #include <playground/cxxmagma/interface/hegv.tcc>
// #include <playground/cxxmagma/interface/hegvx.tcc>
// #include <playground/cxxmagma/interface/herfs.tcc>
// #include <playground/cxxmagma/interface/hesv.tcc>
// #include <playground/cxxmagma/interface/hesvx.tcc>
// #include <playground/cxxmagma/interface/heswapr.tcc>
// #include <playground/cxxmagma/interface/hetd2.tcc>
// #include <playground/cxxmagma/interface/hetf2.tcc>
// #include <playground/cxxmagma/interface/hetrd.tcc>
// #include <playground/cxxmagma/interface/hetrf.tcc>
// #include <playground/cxxmagma/interface/hetri2.tcc>
// #include <playground/cxxmagma/interface/hetri2x.tcc>
// #include <playground/cxxmagma/interface/hetri.tcc>
// #include <playground/cxxmagma/interface/hetrs2.tcc>
// #include <playground/cxxmagma/interface/hetrs.tcc>
// #include <playground/cxxmagma/interface/hfrk.tcc>
// #include <playground/cxxmagma/interface/hgeqz.tcc>
// #include <playground/cxxmagma/interface/hpcon.tcc>
// #include <playground/cxxmagma/interface/hpevd.tcc>
// #include <playground/cxxmagma/interface/hpev.tcc>
// #include <playground/cxxmagma/interface/hpevx.tcc>
// #include <playground/cxxmagma/interface/hpgst.tcc>
// #include <playground/cxxmagma/interface/hpgvd.tcc>
// #include <playground/cxxmagma/interface/hpgv.tcc>
// #include <playground/cxxmagma/interface/hpgvx.tcc>
// #include <playground/cxxmagma/interface/hprfs.tcc>
// #include <playground/cxxmagma/interface/hpsv.tcc>
// #include <playground/cxxmagma/interface/hpsvx.tcc>
// #include <playground/cxxmagma/interface/hptrd.tcc>
// #include <playground/cxxmagma/interface/hptrf.tcc>
// #include <playground/cxxmagma/interface/hptri.tcc>
// #include <playground/cxxmagma/interface/hptrs.tcc>
// #include <playground/cxxmagma/interface/hsein.tcc>
// #include <playground/cxxmagma/interface/hseqr.tcc>
// #include <playground/cxxmagma/interface/ieeeck.tcc>
// #include <playground/cxxmagma/interface/iladlc.tcc>
// #include <playground/cxxmagma/interface/iladlr.tcc>
// #include <playground/cxxmagma/interface/ilalc.tcc>
// #include <playground/cxxmagma/interface/ilalr.tcc>
// #include <playground/cxxmagma/interface/laprec.tcc>
// #include <playground/cxxmagma/interface/ilaslc.tcc>
// #include <playground/cxxmagma/interface/ilaslr.tcc>
// #include <playground/cxxmagma/interface/latrans.tcc>
// #include <playground/cxxmagma/interface/lauplo.tcc>
// #include <playground/cxxmagma/interface/ilaver.tcc>
// #include <playground/cxxmagma/interface/ilazlc.tcc>
// #include <playground/cxxmagma/interface/ilazlr.tcc>
// #include <playground/cxxmagma/interface/interface.tcc>
// #include <playground/cxxmagma/interface/isnan.tcc>
// #include <playground/cxxmagma/interface/izmax1.tcc>
// #include <playground/cxxmagma/interface/labad.tcc>
// #include <playground/cxxmagma/interface/labrd.tcc>
// #include <playground/cxxmagma/interface/lacgv.tcc>
// #include <playground/cxxmagma/interface/lacn2.tcc>
// #include <playground/cxxmagma/interface/lacon.tcc>
// #include <playground/cxxmagma/interface/lacp2.tcc>
// #include <playground/cxxmagma/interface/lacpy.tcc>
// #include <playground/cxxmagma/interface/lacrm.tcc>
// #include <playground/cxxmagma/interface/lacrt.tcc>
// #include <playground/cxxmagma/interface/ladiv.tcc>
// #include <playground/cxxmagma/interface/lae2.tcc>
// #include <playground/cxxmagma/interface/laebz.tcc>
// #include <playground/cxxmagma/interface/laed0.tcc>
// #include <playground/cxxmagma/interface/laed1.tcc>
// #include <playground/cxxmagma/interface/laed2.tcc>
// #include <playground/cxxmagma/interface/laed3.tcc>
// #include <playground/cxxmagma/interface/laed4.tcc>
// #include <playground/cxxmagma/interface/laed5.tcc>
// #include <playground/cxxmagma/interface/laed6.tcc>
// #include <playground/cxxmagma/interface/laed7.tcc>
// #include <playground/cxxmagma/interface/laed8.tcc>
// #include <playground/cxxmagma/interface/laed9.tcc>
// #include <playground/cxxmagma/interface/laeda.tcc>
// #include <playground/cxxmagma/interface/laein.tcc>
// #include <playground/cxxmagma/interface/laesy.tcc>
// #include <playground/cxxmagma/interface/laev2.tcc>
// #include <playground/cxxmagma/interface/laexc.tcc>
// #include <playground/cxxmagma/interface/lag2c.tcc>
// #include <playground/cxxmagma/interface/lag2d.tcc>
// #include <playground/cxxmagma/interface/lag2.tcc>
// #include <playground/cxxmagma/interface/lag2s.tcc>
// #include <playground/cxxmagma/interface/lag2z.tcc>
// #include <playground/cxxmagma/interface/la_gbamv.tcc>
// #include <playground/cxxmagma/interface/la_gbrcond_c.tcc>
// #include <playground/cxxmagma/interface/la_gbrcond.tcc>
// #include <playground/cxxmagma/interface/la_gbrcond_x.tcc>
// #include <playground/cxxmagma/interface/la_gbrpvgrw.tcc>
// #include <playground/cxxmagma/interface/la_geamv.tcc>
// #include <playground/cxxmagma/interface/la_gercond_c.tcc>
// #include <playground/cxxmagma/interface/la_gercond.tcc>
// #include <playground/cxxmagma/interface/la_gercond_x.tcc>
// #include <playground/cxxmagma/interface/lags2.tcc>
// #include <playground/cxxmagma/interface/lagtf.tcc>
// #include <playground/cxxmagma/interface/lagtm.tcc>
// #include <playground/cxxmagma/interface/lagts.tcc>
// #include <playground/cxxmagma/interface/lagv2.tcc>
// #include <playground/cxxmagma/interface/lahef.tcc>
// #include <playground/cxxmagma/interface/la_heramv.tcc>
// #include <playground/cxxmagma/interface/la_hercond_c.tcc>
// #include <playground/cxxmagma/interface/la_hercond_x.tcc>
// #include <playground/cxxmagma/interface/la_herpvgrw.tcc>
// #include <playground/cxxmagma/interface/lahqr.tcc>
// #include <playground/cxxmagma/interface/lahr2.tcc>
// #include <playground/cxxmagma/interface/lahrd.tcc>
// #include <playground/cxxmagma/interface/laic1.tcc>
// #include <playground/cxxmagma/interface/laisnan.tcc>
// #include <playground/cxxmagma/interface/la_lin_berr.tcc>
// #include <playground/cxxmagma/interface/laln2.tcc>
// #include <playground/cxxmagma/interface/lals0.tcc>
// #include <playground/cxxmagma/interface/lalsa.tcc>
// #include <playground/cxxmagma/interface/lalsd.tcc>
// #include <playground/cxxmagma/interface/lamch.tcc>
// #include <playground/cxxmagma/interface/lamrg.tcc>
// #include <playground/cxxmagma/interface/laneg.tcc>
// #include <playground/cxxmagma/interface/langb.tcc>
// #include <playground/cxxmagma/interface/lange.tcc>
// #include <playground/cxxmagma/interface/langt.tcc>
// #include <playground/cxxmagma/interface/lanhb.tcc>
// #include <playground/cxxmagma/interface/lanhe.tcc>
// #include <playground/cxxmagma/interface/lanhf.tcc>
// #include <playground/cxxmagma/interface/lanhp.tcc>
// #include <playground/cxxmagma/interface/lanhs.tcc>
// #include <playground/cxxmagma/interface/lanht.tcc>
// #include <playground/cxxmagma/interface/lansb.tcc>
// #include <playground/cxxmagma/interface/lansf.tcc>
// #include <playground/cxxmagma/interface/lansp.tcc>
// #include <playground/cxxmagma/interface/lanst.tcc>
// #include <playground/cxxmagma/interface/lansy.tcc>
// #include <playground/cxxmagma/interface/lantb.tcc>
// #include <playground/cxxmagma/interface/lantp.tcc>
// #include <playground/cxxmagma/interface/lantr.tcc>
// #include <playground/cxxmagma/interface/lanv2.tcc>
// #include <playground/cxxmagma/interface/lapll.tcc>
// #include <playground/cxxmagma/interface/lapmr.tcc>
// #include <playground/cxxmagma/interface/lapmt.tcc>
// #include <playground/cxxmagma/interface/la_porcond_c.tcc>
// #include <playground/cxxmagma/interface/la_porcond.tcc>
// #include <playground/cxxmagma/interface/la_porcond_x.tcc>
// #include <playground/cxxmagma/interface/la_porpvgrw.tcc>
// #include <playground/cxxmagma/interface/lapy2.tcc>
// #include <playground/cxxmagma/interface/lapy3.tcc>
// #include <playground/cxxmagma/interface/laqgb.tcc>
// #include <playground/cxxmagma/interface/laqge.tcc>
// #include <playground/cxxmagma/interface/laqhb.tcc>
// #include <playground/cxxmagma/interface/laqhe.tcc>
// #include <playground/cxxmagma/interface/laqhp.tcc>
// #include <playground/cxxmagma/interface/laqp2.tcc>
// #include <playground/cxxmagma/interface/laqps.tcc>
// #include <playground/cxxmagma/interface/laqr0.tcc>
// #include <playground/cxxmagma/interface/laqr1.tcc>
// #include <playground/cxxmagma/interface/laqr2.tcc>
// #include <playground/cxxmagma/interface/laqr3.tcc>
// #include <playground/cxxmagma/interface/laqr4.tcc>
// #include <playground/cxxmagma/interface/laqr5.tcc>
// #include <playground/cxxmagma/interface/laqsb.tcc>
// #include <playground/cxxmagma/interface/laqsp.tcc>
// #include <playground/cxxmagma/interface/laqsy.tcc>
// #include <playground/cxxmagma/interface/laqtr.tcc>
// #include <playground/cxxmagma/interface/lar1v.tcc>
// #include <playground/cxxmagma/interface/lar2v.tcc>
// #include <playground/cxxmagma/interface/larcm.tcc>
// #include <playground/cxxmagma/interface/larfb.tcc>
// #include <playground/cxxmagma/interface/larfg.tcc>
// #include <playground/cxxmagma/interface/larfgp.tcc>
// #include <playground/cxxmagma/interface/larf.tcc>
// #include <playground/cxxmagma/interface/larft.tcc>
// #include <playground/cxxmagma/interface/larfx.tcc>
// #include <playground/cxxmagma/interface/largv.tcc>
// #include <playground/cxxmagma/interface/larnv.tcc>
// #include <playground/cxxmagma/interface/la_rpvgrw.tcc>
// #include <playground/cxxmagma/interface/larra.tcc>
// #include <playground/cxxmagma/interface/larrb.tcc>
// #include <playground/cxxmagma/interface/larrc.tcc>
// #include <playground/cxxmagma/interface/larrd.tcc>
// #include <playground/cxxmagma/interface/larre.tcc>
// #include <playground/cxxmagma/interface/larrf.tcc>
// #include <playground/cxxmagma/interface/larrj.tcc>
// #include <playground/cxxmagma/interface/larrk.tcc>
// #include <playground/cxxmagma/interface/larrr.tcc>
// #include <playground/cxxmagma/interface/larrv.tcc>
// #include <playground/cxxmagma/interface/larscl2.tcc>
// #include <playground/cxxmagma/interface/lartg.tcc>
// #include <playground/cxxmagma/interface/lartgp.tcc>
// #include <playground/cxxmagma/interface/lartgs.tcc>
// #include <playground/cxxmagma/interface/lartv.tcc>
// #include <playground/cxxmagma/interface/laruv.tcc>
// #include <playground/cxxmagma/interface/larzb.tcc>
// #include <playground/cxxmagma/interface/larz.tcc>
// #include <playground/cxxmagma/interface/larzt.tcc>
// #include <playground/cxxmagma/interface/las2.tcc>
// #include <playground/cxxmagma/interface/lascl2.tcc>
// #include <playground/cxxmagma/interface/lascl.tcc>
// #include <playground/cxxmagma/interface/lasd0.tcc>
// #include <playground/cxxmagma/interface/lasd1.tcc>
// #include <playground/cxxmagma/interface/lasd2.tcc>
// #include <playground/cxxmagma/interface/lasd3.tcc>
// #include <playground/cxxmagma/interface/lasd4.tcc>
// #include <playground/cxxmagma/interface/lasd5.tcc>
// #include <playground/cxxmagma/interface/lasd6.tcc>
// #include <playground/cxxmagma/interface/lasd7.tcc>
// #include <playground/cxxmagma/interface/lasd8.tcc>
// #include <playground/cxxmagma/interface/lasda.tcc>
// #include <playground/cxxmagma/interface/lasdq.tcc>
// #include <playground/cxxmagma/interface/lasdt.tcc>
// #include <playground/cxxmagma/interface/laset.tcc>
// #include <playground/cxxmagma/interface/lasq1.tcc>
// #include <playground/cxxmagma/interface/lasq2.tcc>
// #include <playground/cxxmagma/interface/lasq3.tcc>
// #include <playground/cxxmagma/interface/lasq4.tcc>
// #include <playground/cxxmagma/interface/lasq5.tcc>
// #include <playground/cxxmagma/interface/lasq6.tcc>
// #include <playground/cxxmagma/interface/lasr.tcc>
// #include <playground/cxxmagma/interface/lasrt.tcc>
// #include <playground/cxxmagma/interface/lassq.tcc>
// #include <playground/cxxmagma/interface/lasv2.tcc>
// #include <playground/cxxmagma/interface/laswp.tcc>
// #include <playground/cxxmagma/interface/lasy2.tcc>
// #include <playground/cxxmagma/interface/la_syamv.tcc>
// #include <playground/cxxmagma/interface/lasyf.tcc>
// #include <playground/cxxmagma/interface/la_syrcond_c.tcc>
// #include <playground/cxxmagma/interface/la_syrcond.tcc>
// #include <playground/cxxmagma/interface/la_syrcond_x.tcc>
// #include <playground/cxxmagma/interface/la_syrpvgrw.tcc>
// #include <playground/cxxmagma/interface/lat2c.tcc>
// #include <playground/cxxmagma/interface/lat2s.tcc>
// #include <playground/cxxmagma/interface/latbs.tcc>
// #include <playground/cxxmagma/interface/latdf.tcc>
// #include <playground/cxxmagma/interface/latps.tcc>
// #include <playground/cxxmagma/interface/latrd.tcc>
// #include <playground/cxxmagma/interface/latrs.tcc>
// #include <playground/cxxmagma/interface/latrz.tcc>
// #include <playground/cxxmagma/interface/latzm.tcc>
// #include <playground/cxxmagma/interface/lauu2.tcc>
// #include <playground/cxxmagma/interface/lauum.tcc>
// #include <playground/cxxmagma/interface/la_wwaddw.tcc>
// #include <playground/cxxmagma/interface/lsame.tcc>
// #include <playground/cxxmagma/interface/lsamen.tcc>
// #include <playground/cxxmagma/interface/opgtr.tcc>
// #include <playground/cxxmagma/interface/opmtr.tcc>
// #include <playground/cxxmagma/interface/orbdb.tcc>
// #include <playground/cxxmagma/interface/orcsd.tcc>
// #include <playground/cxxmagma/interface/org2l.tcc>
// #include <playground/cxxmagma/interface/org2r.tcc>
// #include <playground/cxxmagma/interface/orgbr.tcc>
// #include <playground/cxxmagma/interface/orghr.tcc>
// #include <playground/cxxmagma/interface/orgl2.tcc>
// #include <playground/cxxmagma/interface/orglq.tcc>
// #include <playground/cxxmagma/interface/orgql.tcc>
// #include <playground/cxxmagma/interface/orgqr.tcc>
// #include <playground/cxxmagma/interface/orgr2.tcc>
// #include <playground/cxxmagma/interface/orgrq.tcc>
// #include <playground/cxxmagma/interface/orgtr.tcc>
// #include <playground/cxxmagma/interface/orm2l.tcc>
// #include <playground/cxxmagma/interface/orm2r.tcc>
// #include <playground/cxxmagma/interface/ormbr.tcc>
// #include <playground/cxxmagma/interface/ormhr.tcc>
// #include <playground/cxxmagma/interface/orml2.tcc>
// #include <playground/cxxmagma/interface/ormlq.tcc>
#include <playground/cxxmagma/interface/ormql.tcc>
#include <playground/cxxmagma/interface/ormqr.tcc>
// #include <playground/cxxmagma/interface/ormr2.tcc>
// #include <playground/cxxmagma/interface/ormr3.tcc>
// #include <playground/cxxmagma/interface/ormrq.tcc>
// #include <playground/cxxmagma/interface/ormrz.tcc>
// #include <playground/cxxmagma/interface/ormtr.tcc>
// #include <playground/cxxmagma/interface/pbcon.tcc>
// #include <playground/cxxmagma/interface/pbequ.tcc>
// #include <playground/cxxmagma/interface/pbrfs.tcc>
// #include <playground/cxxmagma/interface/pbstf.tcc>
// #include <playground/cxxmagma/interface/pbsv.tcc>
// #include <playground/cxxmagma/interface/pbsvx.tcc>
// #include <playground/cxxmagma/interface/pbtf2.tcc>
// #include <playground/cxxmagma/interface/pbtrf.tcc>
// #include <playground/cxxmagma/interface/pbtrs.tcc>
// #include <playground/cxxmagma/interface/pftrf.tcc>
// #include <playground/cxxmagma/interface/pftri.tcc>
// #include <playground/cxxmagma/interface/pftrs.tcc>
// #include <playground/cxxmagma/interface/pocon.tcc>
// #include <playground/cxxmagma/interface/poequb.tcc>
// #include <playground/cxxmagma/interface/poequ.tcc>
// #include <playground/cxxmagma/interface/porfs.tcc>
#include <playground/cxxmagma/interface/posv.tcc>
// #include <playground/cxxmagma/interface/posvx.tcc>
// #include <playground/cxxmagma/interface/potf2.tcc>
#include <playground/cxxmagma/interface/potrf.tcc>
#include <playground/cxxmagma/interface/potri.tcc>
// #include <playground/cxxmagma/interface/potrs.tcc>
// #include <playground/cxxmagma/interface/ppcon.tcc>
// #include <playground/cxxmagma/interface/ppequ.tcc>
// #include <playground/cxxmagma/interface/pprfs.tcc>
// #include <playground/cxxmagma/interface/ppsv.tcc>
// #include <playground/cxxmagma/interface/ppsvx.tcc>
// #include <playground/cxxmagma/interface/pptrf.tcc>
// #include <playground/cxxmagma/interface/pptri.tcc>
// #include <playground/cxxmagma/interface/pptrs.tcc>
// #include <playground/cxxmagma/interface/pstf2.tcc>
// #include <playground/cxxmagma/interface/pstrf.tcc>
// #include <playground/cxxmagma/interface/ptcon.tcc>
// #include <playground/cxxmagma/interface/pteqr.tcc>
// #include <playground/cxxmagma/interface/ptrfs.tcc>
// #include <playground/cxxmagma/interface/ptsv.tcc>
// #include <playground/cxxmagma/interface/ptsvx.tcc>
// #include <playground/cxxmagma/interface/pttrf.tcc>
// #include <playground/cxxmagma/interface/pttrs.tcc>
// #include <playground/cxxmagma/interface/ptts2.tcc>
// #include <playground/cxxmagma/interface/rot.tcc>
// #include <playground/cxxmagma/interface/rscl.tcc>
// #include <playground/cxxmagma/interface/sbevd.tcc>
// #include <playground/cxxmagma/interface/sbev.tcc>
// #include <playground/cxxmagma/interface/sbevx.tcc>
// #include <playground/cxxmagma/interface/sbgst.tcc>
// #include <playground/cxxmagma/interface/sbgvd.tcc>
// #include <playground/cxxmagma/interface/sbgv.tcc>
// #include <playground/cxxmagma/interface/sbgvx.tcc>
// #include <playground/cxxmagma/interface/sbtrd.tcc>
// #include <playground/cxxmagma/interface/sfrk.tcc>
// #include <playground/cxxmagma/interface/sgesv.tcc>
// #include <playground/cxxmagma/interface/spevd.tcc>
// #include <playground/cxxmagma/interface/spev.tcc>
// #include <playground/cxxmagma/interface/spevx.tcc>
// #include <playground/cxxmagma/interface/spgst.tcc>
// #include <playground/cxxmagma/interface/spgvd.tcc>
// #include <playground/cxxmagma/interface/spgv.tcc>
// #include <playground/cxxmagma/interface/spgvx.tcc>
// #include <playground/cxxmagma/interface/spmv.tcc>
// #include <playground/cxxmagma/interface/sposv.tcc>
// #include <playground/cxxmagma/interface/sprfs.tcc>
// #include <playground/cxxmagma/interface/spr.tcc>
// #include <playground/cxxmagma/interface/spsv.tcc>
// #include <playground/cxxmagma/interface/spsvx.tcc>
// #include <playground/cxxmagma/interface/sptrd.tcc>
// #include <playground/cxxmagma/interface/sptrf.tcc>
// #include <playground/cxxmagma/interface/sptri.tcc>
// #include <playground/cxxmagma/interface/sptrs.tcc>
// #include <playground/cxxmagma/interface/stebz.tcc>
// #include <playground/cxxmagma/interface/stedc.tcc>
// #include <playground/cxxmagma/interface/stegr.tcc>
// #include <playground/cxxmagma/interface/stein.tcc>
// #include <playground/cxxmagma/interface/stemr.tcc>
// #include <playground/cxxmagma/interface/steqr.tcc>
// #include <playground/cxxmagma/interface/sterf.tcc>
// #include <playground/cxxmagma/interface/stevd.tcc>
// #include <playground/cxxmagma/interface/stev.tcc>
// #include <playground/cxxmagma/interface/stevr.tcc>
// #include <playground/cxxmagma/interface/stevx.tcc>
// #include <playground/cxxmagma/interface/sycon.tcc>
// #include <playground/cxxmagma/interface/syconv.tcc>
// #include <playground/cxxmagma/interface/syequb.tcc>
// #include <playground/cxxmagma/interface/syevd.tcc>
// #include <playground/cxxmagma/interface/syev.tcc>
// #include <playground/cxxmagma/interface/syevr.tcc>
// #include <playground/cxxmagma/interface/syevx.tcc>
// #include <playground/cxxmagma/interface/sygs2.tcc>
// #include <playground/cxxmagma/interface/sygst.tcc>
// #include <playground/cxxmagma/interface/sygvd.tcc>
// #include <playground/cxxmagma/interface/sygv.tcc>
// #include <playground/cxxmagma/interface/sygvx.tcc>
// #include <playground/cxxmagma/interface/symv.tcc>
// #include <playground/cxxmagma/interface/syrfs.tcc>
// #include <playground/cxxmagma/interface/syr.tcc>
// #include <playground/cxxmagma/interface/sysv.tcc>
// #include <playground/cxxmagma/interface/sysvx.tcc>
// #include <playground/cxxmagma/interface/syswapr.tcc>
// #include <playground/cxxmagma/interface/sytd2.tcc>
// #include <playground/cxxmagma/interface/sytf2.tcc>
// #include <playground/cxxmagma/interface/sytrd.tcc>
// #include <playground/cxxmagma/interface/sytrf.tcc>
// #include <playground/cxxmagma/interface/sytri2.tcc>
// #include <playground/cxxmagma/interface/sytri2x.tcc>
// #include <playground/cxxmagma/interface/sytri.tcc>
// #include <playground/cxxmagma/interface/sytrs2.tcc>
// #include <playground/cxxmagma/interface/sytrs.tcc>
// #include <playground/cxxmagma/interface/tbcon.tcc>
// #include <playground/cxxmagma/interface/tbrfs.tcc>
// #include <playground/cxxmagma/interface/tbtrs.tcc>
// #include <playground/cxxmagma/interface/tfsm.tcc>
// #include <playground/cxxmagma/interface/tftri.tcc>
// #include <playground/cxxmagma/interface/tfttp.tcc>
// #include <playground/cxxmagma/interface/tfttr.tcc>
// #include <playground/cxxmagma/interface/tgevc.tcc>
// #include <playground/cxxmagma/interface/tgex2.tcc>
// #include <playground/cxxmagma/interface/tgexc.tcc>
// #include <playground/cxxmagma/interface/tgsen.tcc>
// #include <playground/cxxmagma/interface/tgsja.tcc>
// #include <playground/cxxmagma/interface/tgsna.tcc>
// #include <playground/cxxmagma/interface/tgsy2.tcc>
// #include <playground/cxxmagma/interface/tgsyl.tcc>
// #include <playground/cxxmagma/interface/tpcon.tcc>
// #include <playground/cxxmagma/interface/tprfs.tcc>
// #include <playground/cxxmagma/interface/tptri.tcc>
// #include <playground/cxxmagma/interface/tptrs.tcc>
// #include <playground/cxxmagma/interface/tpttf.tcc>
// #include <playground/cxxmagma/interface/tpttr.tcc>
// #include <playground/cxxmagma/interface/trcon.tcc>
// #include <playground/cxxmagma/interface/trevc.tcc>
// #include <playground/cxxmagma/interface/trexc.tcc>
// #include <playground/cxxmagma/interface/trrfs.tcc>
// #include <playground/cxxmagma/interface/trsen.tcc>
// #include <playground/cxxmagma/interface/trsna.tcc>
// #include <playground/cxxmagma/interface/trsyl.tcc>
// #include <playground/cxxmagma/interface/trti2.tcc>
// #include <playground/cxxmagma/interface/trtri.tcc>
// #include <playground/cxxmagma/interface/trtrs.tcc>
// #include <playground/cxxmagma/interface/trttf.tcc>
// #include <playground/cxxmagma/interface/trttp.tcc>
// #include <playground/cxxmagma/interface/tzrqf.tcc>
// #include <playground/cxxmagma/interface/tzrzf.tcc>
// #include <playground/cxxmagma/interface/unbdb.tcc>
// #include <playground/cxxmagma/interface/uncsd.tcc>
// #include <playground/cxxmagma/interface/ung2l.tcc>
// #include <playground/cxxmagma/interface/ung2r.tcc>
// #include <playground/cxxmagma/interface/ungbr.tcc>
// #include <playground/cxxmagma/interface/unghr.tcc>
// #include <playground/cxxmagma/interface/ungl2.tcc>
// #include <playground/cxxmagma/interface/unglq.tcc>
// #include <playground/cxxmagma/interface/ungql.tcc>
// #include <playground/cxxmagma/interface/ungqr.tcc>
// #include <playground/cxxmagma/interface/ungr2.tcc>
// #include <playground/cxxmagma/interface/ungrq.tcc>
// #include <playground/cxxmagma/interface/ungtr.tcc>
// #include <playground/cxxmagma/interface/unm2l.tcc>
// #include <playground/cxxmagma/interface/unm2r.tcc>
// #include <playground/cxxmagma/interface/unmbr.tcc>
// #include <playground/cxxmagma/interface/unmhr.tcc>
// #include <playground/cxxmagma/interface/unml2.tcc>
// #include <playground/cxxmagma/interface/unmlq.tcc>
#include <playground/cxxmagma/interface/unmql.tcc>
#include <playground/cxxmagma/interface/unmqr.tcc>
// #include <playground/cxxmagma/interface/unmr2.tcc>
// #include <playground/cxxmagma/interface/unmr3.tcc>
// #include <playground/cxxmagma/interface/unmrq.tcc>
// #include <playground/cxxmagma/interface/unmrz.tcc>
// #include <playground/cxxmagma/interface/unmtr.tcc>
// #include <playground/cxxmagma/interface/upgtr.tcc>
// #include <playground/cxxmagma/interface/upmtr.tcc>
// #include <playground/cxxmagma/interface/zsum1.tcc>

#endif // PLAYGROUND_CXXMAGMA_INTERFACE_INTERFACE_TCC