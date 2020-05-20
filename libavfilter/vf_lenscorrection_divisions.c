/*
 * Copyright (C) 2007 Richard Spindler (author of frei0r plugin from which this was derived)
 * Copyright (C) 2014 Daniel Oberhoff
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Lenscorrection_Divisions filter, algorithm from the frei0r plugin with the same name
*/
#include <stdlib.h>
#include <math.h>

#include "libavutil/opt.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"

#include "avfilter.h"
#include "internal.h"
#include "video.h"

#define INLINE av_always_inline

#define LUT_SIZE 256
#define LUT_U(i) (i * 2)
#define LUT_R(i) (i * 2 + 1)

static INLINE double min(double a, double b) { return (a <= b) ? a : b; }
static INLINE double max(double a, double b) { return (a >= b) ? a : b; }

static INLINE double lerp(double a, double b, double t) { return (1.0 - t) * a + t * b; }

static INLINE double distort_divisions_math(double k1, double k2, double r)
{
    //av_assert1(r >= 0);
    const double r2 = r * r;
    const double r4 = r2 * r2;
    const double div = (1 + k1 * r2 + k2 * r4);
    /* if (div <= 0.0000001) { */
    /*   av_log( */
    /*       NULL, AV_LOG_ERROR, */
    /*       "lenscorrection_divisions: your k1 (%f) and k2 (%f) are weird, (1. + k1 * r2 " */
    /*       "+ k2 * r4) is zero or negative at %f: %f\n", */
    /*       k1, k2, div, r); */
    /*   return 1; */
    /* } */
    const double u = r / div;
    //const double u = r * div;
    //return pow(r, 1 + k1);
    return u;
}

static INLINE int small(double x, double eps) {
    return fabs(x) <= eps;
    //return x <= eps && -x <= eps;
}

static INLINE double distort(
    double k1,
    double k2,
    double target_u,
    double *ulast_arg,
    double rlast_arg,
    double maxerr
    )
{
    double u = *ulast_arg;
    double r = rlast_arg;

    double r_u_ratio = r / u;
    double target_delta = (target_u - u);

    int i = 0;
    do {
        // assume relation is linear and step towards presumed target
        const double diff_ratio = target_delta * r_u_ratio;
        r += diff_ratio;

        u = distort_divisions_math(k1, k2, r);

/*        if (r < 0 || u > 4) {
            av_log(NULL, AV_LOG_INFO, "%d. target_u: %f, target_delta: %f, u: %f, r: %f: \n", i, target_u, target_delta, u, r);
            while (1) {}
        }
*/
        r_u_ratio = r / u;
        target_delta = (target_u - u);

/*        if (abs(r - u) > 0.01) {
            av_log(NULL, AV_LOG_INFO, "%d: %f -> %f, %f\n", i, k1, k2, u, r, target_delta);
        }
*/
        // error is small enough, return
        if (small(target_delta, maxerr)) {
/*            const int N = 16;
            if (i > N) {
                //av_log(NULL, AV_LOG_INFO, "MORE THAN %d steps: %d. target_u: %f, rlast: %f, target_delta: %f, u: %f, ulast: %f, r: %f, rlast: %f: \n", N, i, target_u, rlast, target_delta, u, ulast, r, rlast);
                }
*/
            *ulast_arg = u;
            return r;
        }

        ++i;
    } while (1);
}

typedef struct LenscorrectionCtx {
    const AVClass *av_class;
    int width;
    int height;
    int hsub, vsub;
    int nb_planes;
    double cx, cy, k1, k2;
    int outw, outh;
    double maxerr, rstart;
    double outer_circle;
    double umax;
    double* distortion_lut;
} LenscorrectionCtx;

#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
static const AVOption lenscorrection_divisions_options[] = {
    { "cx",     "set relative center x", offsetof(LenscorrectionCtx, cx), AV_OPT_TYPE_DOUBLE, {.dbl=0.5}, 0, 1, .flags=FLAGS },
    { "cy",     "set relative center y", offsetof(LenscorrectionCtx, cy), AV_OPT_TYPE_DOUBLE, {.dbl=0.5}, 0, 1, .flags=FLAGS },
    { "k1",     "set quadratic distortion factor", offsetof(LenscorrectionCtx, k1), AV_OPT_TYPE_DOUBLE, {.dbl=0.0}, -1, 1, .flags=FLAGS },
    { "k2",     "set double quadratic distortion factor", offsetof(LenscorrectionCtx, k2), AV_OPT_TYPE_DOUBLE, {.dbl=0.0}, -1, 1, .flags=FLAGS },
    { "outw",   "set output width", offsetof(LenscorrectionCtx, outw), AV_OPT_TYPE_INT, {.dbl=0}, 0, 32768, .flags=FLAGS },
    { "outh",   "set output height", offsetof(LenscorrectionCtx, outh), AV_OPT_TYPE_INT, {.dbl=0}, 0, 32768, .flags=FLAGS },
    { "maxerr",   "set distortion maximum error", offsetof(LenscorrectionCtx, maxerr), AV_OPT_TYPE_DOUBLE, {.dbl=0.0001}, 0, 0.1, .flags=FLAGS },
    // empirically / guesstimated, but doesn't seem to matter much
    { "rstart",   "set starting guess for distorted radius", offsetof(LenscorrectionCtx, rstart), AV_OPT_TYPE_DOUBLE, {.dbl=1.0}, 0, 2, .flags=FLAGS },
    { "outer_circle",   "what circle with r=1 is aligned to. vf_lenscorrection uses outer circle (1), blender/imagemagick inner circle (0)", offsetof(LenscorrectionCtx, outer_circle), AV_OPT_TYPE_DOUBLE, {.dbl=0}, 0, 1, .flags=FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(lenscorrection_divisions);

typedef struct ThreadData {
    AVFrame *in, *out;
    int w, h;
    int outw, outh;
    int plane;
    double xcenter, ycenter;
    double outxcenter, outycenter;
    double k1, k2;
    double r_px_1;
    double maxerr, rstart, umax;
    double* distortion_lut;
} ThreadData;

static INLINE uint8_t sample(const int x, const int y, const int w, const int h, const int inlinesize, const uint8_t* indata)
{
    const char isvalid = x >= 0 && x < w - 1 && y >= 0 && y < h - 1;
    uint8_t c = isvalid ? indata[y * inlinesize + x] : 0;
    return c;
}

static int filter_slice(AVFilterContext *ctx, void *arg, int job, int nb_jobs)
{
    ThreadData *td = arg;
    AVFrame *in = td->in;
    AVFrame *out = td->out;

    const double k1 = td->k1;
    const double k2 = td->k2;
    const double maxerr = td->maxerr;
    const double umax = td->umax;
    const double rstart = td->rstart;

    const double* distortion_lut = td->distortion_lut;

    const int w = td->w;
    const int h = td->h;
    const int outw = td->outw;
    const int outh = td->outh;
    const double xcenter = td->xcenter;
    const double ycenter = td->ycenter;

    // half pixel offset because our x/y loop is over integers
    const double outxcenter = td->outxcenter - 0.5;
    const double outycenter = td->outycenter - 0.5;
    const double r_px_1 = td->r_px_1;
    const double r_px_1_inv = 1.0 / r_px_1;

    const int starty = (outh *  job   ) / nb_jobs;
    const int endy   = (outh * (job+1)) / nb_jobs;

    const int plane = td->plane;
    const int inlinesize = in->linesize[plane];
    const int outlinesize = out->linesize[plane];
    const uint8_t *indata = in->data[plane];
    uint8_t *outrow = out->data[plane] + starty * outlinesize;

    double u_r1_inv = 1 / distort_divisions_math(k1, k2, 1.0);

    // start at u ~= 1
    //double rlast = 0.8;
    double r = rstart;
    double ulast = distort_divisions_math(k1, k2, r);

    double ustep_inv = (LUT_SIZE - 1) / umax;

    for (int outy = starty; outy < endy; outy++, outrow += outlinesize) {
        const double off_y = outy - outycenter;
        const double off_y2 = off_y * off_y;
        uint8_t *out = outrow;
        for (int outx = 0; outx < outw; outx++) {
            const double off_x = outx - outxcenter;
            const double off_x2 = off_x * off_x;
            const double u_px = sqrt(off_x2 + off_y2);
            const double u = u_px * r_px_1_inv;
            uint8_t c = 0;
            int i = u * ustep_inv - 0.5;
            if (i < LUT_SIZE || 1) {
                double u0 = 0;
                double r0 = 0;
                if (i > 0 && i < LUT_SIZE) {
                  u0 = distortion_lut[LUT_U(i - 1)];
                  r0 = distortion_lut[LUT_R(i - 1)];
                }
                const double u1 = distortion_lut[LUT_U(i)];
                const double r1 = distortion_lut[LUT_R(i)];
                const double t = (u - u0) / (u1 - u0);

                if ((outx / 64) % 2 == 0) {
                  //r = u / u1 * r1;
                  //r = r1;
                  //r = distort(k1, k2, u, &ulast, r, maxerr);
                }
                //else
                {
                  //r = lerp(r0, r1, t);
                  r = distort_divisions_math(k1, k2, u);
                  //r = pow(u, 1/(1 + k1));
                  ulast = u;
                }
                //r = distort_divisions_math(-k1, -k2, u);
                /*            if (fabs(u - 1.0) < 0.0001) {
                                av_log(NULL, AV_LOG_INFO, "u: %f, r: %f => %f
                   --- %f\n", u, r, r * r1_u_ratio, r1_u_ratio);
                            }
                */
                /* if (fabs(u - (r / (1 + k1 * r * r + k2 * r * r * r * r)) > 0.01)) { */
                /*     av_log(NULL, AV_LOG_INFO, "WTFFFFFFFFFFFFFFFFFFFF u:%f, r: %f\n", u, r); */
                /* } */
                //normalize vector length
                r /= u;
                const double xf = xcenter + (off_x * r);
                const double yf = ycenter + (off_y * r);
                double x_low = floor(xf) + 0.5;
                double y_low = floor(yf) + 0.5;
                if (xf < x_low) { x_low -= 1; }
                if (yf < y_low) { y_low -= 1; }

                const double tx = xf - x_low;
                const double ty = yf - y_low;

                // bilinear
                const int x = (int)(x_low);
                const int y = (int)(y_low);
                const uint8_t c00 = sample(x, y, w, h, inlinesize, indata);
                const uint8_t c10 = sample(x + 1, y, w, h, inlinesize, indata);
                const uint8_t c01 = sample(x, y + 1, w, h, inlinesize, indata);
                const uint8_t c11 = sample(x + 1, y + 1, w, h, inlinesize, indata);

                const uint8_t cx0 = lerp(c00, c10, tx);
                const uint8_t cx1 = lerp(c01, c11, tx);
                c = lerp(cx0, cx1, ty);

            }
            else {
                //av_log(NULL, AV_LOG_INFO, "u >= umax: %f %f\n", u, umax);
            }
/*            if (fabs(u_px - r_px_1) < maxerr * 10000 && plane == 2) {
                c = 255;
            }
*/
/*            if (fabs(u - 1) < maxerr * 10 && plane == 1) {
                c = 255;
            }
            if (fabs(r - 1) < maxerr * 10 && plane == 0) {
                c = 255;
            }
*/
            *out++ = c;
        }
    }
    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV410P,
        AV_PIX_FMT_YUV444P,  AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUVJ420P,
        AV_PIX_FMT_YUVA444P, AV_PIX_FMT_YUVA420P,
        AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRAP,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static av_cold void uninit(AVFilterContext *ctx)
{
}

static int config_props(AVFilterLink *outlink) {
  AVFilterContext *ctx = outlink->src;
  LenscorrectionCtx *rect = ctx->priv;
  AVFilterLink *inlink = ctx->inputs[0];
  const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(inlink->format);
  rect->hsub = pixdesc->log2_chroma_w;
  rect->vsub = pixdesc->log2_chroma_h;

  //rect->k1 *= -1;
  //rect->k2 *= -1;

  const int w = inlink->w;
  const int h = inlink->h;

  rect->width = w;
  rect->height = h;
  if (rect->outw <= 0) {
    rect->outw = w;
  }
  if (rect->outh <= 0) {
    rect->outh = h;
  }
  outlink->w = rect->outw;
  outlink->h = rect->outh;
  rect->nb_planes = av_pix_fmt_count_planes(inlink->format);


  const int outw = rect->outw;
  const int outh = rect->outh;
  
  const double xcenter = rect->cx * w;
  const double ycenter = rect->cy * h;
  const double out_x_offset = round((outw - w) / 2.);
  const double out_y_offset = round((outh - h) / 2.);
  const double outxcenter = xcenter + out_x_offset;
  const double outycenter = ycenter + out_y_offset;

  // longest distance in distorted/src image
  const double in_xmax = max(xcenter, w - xcenter);
  const double in_ymax = max(ycenter, h - ycenter);
  const double in_max = sqrt(in_xmax * in_xmax + in_ymax * in_ymax);
  const double out_xmax = max(xcenter, w - xcenter);
  const double out_ymax = max(ycenter, h - ycenter);
  const double out_max = sqrt(out_xmax * out_xmax + out_ymax * out_ymax);
  const double rmax_px = max(in_max, out_max);


  // circle at r=1 is centered on center and touches nearest image edge
  // shortest distance from center to image edge
  const double in_xmin = min(xcenter, w - xcenter);
  const double in_ymin = min(ycenter, h - ycenter);
  const double r_px_1_inner = min(in_xmin, in_ymin);
  const double r_px_1_outer = max(in_xmax, in_ymax);
  //const double r_px_1 = lerp(r_px_1_inner, r_px_1_outer, rect->outer_circle);
  const double r_px_1 = r_px_1_inner;
  

  const double rmax = rmax_px / r_px_1;

  const double maxerr = rect->maxerr;

  av_log(NULL, AV_LOG_INFO, "RMAX: %f, %f / %f\n", rmax, rmax_px, r_px_1);
  //find umax
  {
      const double k1 = rect->k1;
      const double k2 = rect->k2;

      double rlast;
      const double step = maxerr;
      double r = step;
      double u = distort_divisions_math(k1, k2, r);
      double ulast = 0;
      while (u > ulast && r < 5  && r < rmax) {
        ulast = u;
        rlast = r;
        r += step;
        u = distort_divisions_math(k1, k2, r);
        //printf("umax: %f %f\n", ulast, rlast);
      }
      const double umax = ulast;
      rect->umax = umax;

      double *lut = av_malloc_array(LUT_SIZE * 2, sizeof(double));
      double u_r1 = distort_divisions_math(k1, k2, 1);

      const double ustep = umax / (LUT_SIZE - 1);
      {
          double rstep = ustep / 32;
          av_log(NULL, AV_LOG_INFO, "%f -> %f, %f -> , %f, %f\n", rmax, umax, rstep, ustep, u_r1);
          double r = 0;
          double u = 0;
          // offset by 1, but set last one manually
          for (int iu = 0; iu < LUT_SIZE; ++iu) {

            const double target_u = (iu + 1) * ustep;
              while (u < (target_u + maxerr) && r < rmax) {
                  r += rstep;
                  u = distort_divisions_math(k1, k2, r);
                  //av_log(NULL, AV_LOG_INFO, "loop(%d): %f -> %f\n", iu, u, r);
              } ;

              lut[LUT_U(iu)] = u ;
              lut[LUT_R(iu)] = r;
          }
          lut[LUT_U(LUT_SIZE - 1)] = umax;
          lut[LUT_R(LUT_SIZE - 1)] = rmax;

      }
      av_log(NULL, AV_LOG_INFO, "i, u, r\n");
      for (int i = 0; i < LUT_SIZE; ++i) {
        //lut[LUT_R(i)] *= u_r1;
          av_log(NULL, AV_LOG_INFO, "%d, %f, %f\n", i, lut[i * 2], lut[i * 2 + 1]);
      }
      av_log(NULL, AV_LOG_INFO, "LUTDONE\n");
      rect->distortion_lut = lut;
  }
  return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    LenscorrectionCtx *rect = (LenscorrectionCtx*)ctx->priv;
    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    int plane;

    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }

    av_frame_copy_props(out, in);

    const double k1 = rect->k1;
    const double k2 = rect->k2;
    const double maxerr = rect->maxerr;
    const double rstart = rect->rstart;
    double umax = rect->umax;
    double* distortion_lut = rect->distortion_lut;

    for (plane = 0; plane < rect->nb_planes; ++plane) {
        const int hsub = plane == 1 || plane == 2 ? rect->hsub : 0;
        const int vsub = plane == 1 || plane == 2 ? rect->vsub : 0;
        const int w = AV_CEIL_RSHIFT(rect->width, hsub);
        const int h = AV_CEIL_RSHIFT(rect->height, vsub);
        const int outw = AV_CEIL_RSHIFT(rect->outw, hsub);
        const int outh = AV_CEIL_RSHIFT(rect->outh, vsub);

        const double xcenter = rect->cx * w;
        const double ycenter = rect->cy * h;
        const double out_x_offset = round((outw - w) / 2.);
        const double out_y_offset = round((outh - h) / 2.);
        const double outxcenter = xcenter + out_x_offset;
        const double outycenter = ycenter + out_y_offset;
        // circle at r=1 is centered on center and touches nearest image edge
        // shortest distance from center to image edge
        const double r_px_1_inner = min(xcenter, min(w - xcenter, min(ycenter, h - ycenter)));
        const double r_px_1_outer = max(xcenter, max(w - xcenter, max(ycenter, h - ycenter)));
        //const double r_px_1 = lerp(r_px_1_inner, r_px_1_outer, rect->outer_circle);
        const double r_px_1 = r_px_1_inner;

        // longest distance in distorted/src image
        const double in_xmax = max(xcenter, w - xcenter);
        const double in_ymax = max(ycenter, h - ycenter);
        const double rmax_px = sqrt(in_xmax * in_xmax + in_ymax * in_ymax);
        umax = min(umax, distort_divisions_math(k1, k2, rmax_px / r_px_1));

        //printf("PLANE %d: %f %f %f\t(%f, %f)(%f, %f)\n", plane, r_px_1, rmax_px, umax, xcenter, ycenter, outxcenter, outycenter);

        ThreadData td = {
            .in = in,
            .out  = out,
            .w  = w,
            .h  = h,
            .k1 = k1,
            .k2 = k2,
            .outw  = outw,
            .outh  = outh,
            .xcenter = xcenter,
            .ycenter = ycenter,
            .outxcenter = outxcenter,
            .outycenter = outycenter,
            .plane = plane,
            .r_px_1 = r_px_1,
            .rstart = rstart,
            .umax = umax,
            .maxerr = maxerr,
            .distortion_lut = distortion_lut
        };

        ctx->internal->execute(ctx, filter_slice, &td, NULL, FFMIN(outh, ff_filter_get_nb_threads(ctx)));
    }

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static const AVFilterPad lenscorrection_divisions_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad lenscorrection_divisions_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
    },
    { NULL }
};

AVFilter ff_vf_lenscorrection_divisions = {
    .name          = "lenscorrection_divisions",
    .description   = NULL_IF_CONFIG_SMALL("Rectify the image by correcting for lens distortion."),
    .priv_size     = sizeof(LenscorrectionCtx),
    .query_formats = query_formats,
    .inputs        = lenscorrection_divisions_inputs,
    .outputs       = lenscorrection_divisions_outputs,
    .priv_class    = &lenscorrection_divisions_class,
    .uninit        = uninit,
    .flags         = AVFILTER_FLAG_SLICE_THREADS,
};
