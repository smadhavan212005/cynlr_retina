// faithful js/opencv.js port of the no_reflection/retinal_opl_v2.py pipeline,
// plus the temporal blending and camera-parameter mapping for live streaming.

export const SIGMA_C_BASE = 1.5;
export const SIGMA_S_BASE = 6.0;

export const DIM_MEAN_LUMINANCE_THRESHOLD = 60;
export const SCOTOPIC_MEAN_LUMINANCE_FLOOR = 20;
export const ROD_POOLING_SIGMA = 1.2;
export const TAU_SCOTOPIC_MAX = 0.20;
export const DOG_NOISE_MAD_MULTIPLIER = 4.0;

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// mean_luminance
export function meanLuminance(cv, luminance) {
  return cv.mean(luminance)[0];
}

// is_dim_frame
export function isDimFrame(meanLum, threshold = DIM_MEAN_LUMINANCE_THRESHOLD) {
  return meanLum < threshold;
}

// dimness_fraction_for_scotopic_taper
export function dimnessFraction(meanLum, brightThresh = DIM_MEAN_LUMINANCE_THRESHOLD, darkFloor = SCOTOPIC_MEAN_LUMINANCE_FLOOR) {
  if (meanLum >= brightThresh) return 0;
  if (meanLum <= darkFloor) return 1;
  return (brightThresh - meanLum) / (brightThresh - darkFloor);
}

// scotopic_temporal_integration_tau_taper
export function scotopicTau(tauBase, meanLum, tauScotopicMax = TAU_SCOTOPIC_MAX) {
  const frac = dimnessFraction(meanLum);
  return tauBase + frac * (tauScotopicMax - tauBase);
}

// rgb_to_luminance
export function rgbToLuminance(cv, srcBgr) {
  const chans = new cv.MatVector();
  cv.split(srcBgr, chans);
  const bChan = chans.get(0), gChan = chans.get(1), rChan = chans.get(2);
  const bf = new cv.Mat(), gf = new cv.Mat(), rf = new cv.Mat();
  bChan.convertTo(bf, cv.CV_32F);
  gChan.convertTo(gf, cv.CV_32F);
  rChan.convertTo(rf, cv.CV_32F);
  const luminance = new cv.Mat();
  cv.addWeighted(rf, 0.299, gf, 0.587, 0, luminance);
  cv.addWeighted(luminance, 1, bf, 0.114, 0, luminance);
  bChan.delete(); gChan.delete(); rChan.delete();
  chans.delete(); bf.delete(); gf.delete(); rf.delete();
  return luminance;
}

// log_transform
export function logTransform(cv, luminance, c = null) {
  const ones = new cv.Mat(luminance.rows, luminance.cols, luminance.type(), new cv.Scalar(1));
  const onePlus = new cv.Mat();
  cv.add(luminance, ones, onePlus);
  ones.delete();
  const logImg = new cv.Mat();
  cv.log(onePlus, logImg);
  onePlus.delete();

  let cVal = c;
  if (cVal === null) {
    const maxVal = cv.minMaxLoc(luminance).maxVal;
    cVal = 255.0 / Math.log(1 + maxVal);
  }
  const result = new cv.Mat();
  logImg.convertTo(result, -1, cVal, 0);
  logImg.delete();
  return result;
}

// rod_convergence_pooling
export function rodPooling(cv, luminance) {
  return gaussianBlur(cv, luminance, ROD_POOLING_SIGMA);
}

// noise_sigma_via_median_absolute_deviation
export function estimateNoiseSigma(dog) {
  const data = dog.data32F;
  const n = data.length;
  const sorted = Float32Array.from(data).sort();
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[(n - 1) / 2];
  const absDev = new Float32Array(n);
  for (let i = 0; i < n; i++) absDev[i] = Math.abs(data[i] - median);
  absDev.sort();
  const mad = n % 2 === 0 ? (absDev[n / 2 - 1] + absDev[n / 2]) / 2 : absDev[(n - 1) / 2];
  return 1.4826 * mad;
}

// kernel_size_from_sigma + apply_gaussian_blur
export function gaussianBlur(cv, image, sigma) {
  const ksize = Math.floor(2 * Math.ceil(3 * sigma) + 1);
  const dst = new cv.Mat();
  cv.GaussianBlur(image, dst, new cv.Size(ksize, ksize), sigma, sigma, cv.BORDER_DEFAULT);
  return dst;
}

// centre_minus_surround
export function differenceOfGaussians(cv, image, sigmaC, sigmaS) {
  const centre = gaussianBlur(cv, image, sigmaC);
  const surround = gaussianBlur(cv, image, sigmaS);
  const dog = new cv.Mat();
  cv.subtract(centre, surround, dog);
  centre.delete(); surround.delete();
  return dog;
}

// half_wave_rectify_positive
export function rectifyOn(cv, dog) {
  const zero = new cv.Mat(dog.rows, dog.cols, dog.type(), new cv.Scalar(0));
  const out = new cv.Mat();
  cv.max(dog, zero, out);
  zero.delete();
  return out;
}

// half_wave_rectify_negative
export function rectifyOff(cv, dog) {
  const neg = new cv.Mat();
  dog.convertTo(neg, -1, -1, 0);
  const zero = new cv.Mat(dog.rows, dog.cols, dog.type(), new cv.Scalar(0));
  const out = new cv.Mat();
  cv.max(neg, zero, out);
  neg.delete(); zero.delete();
  return out;
}

// scale_to_0_255
export function normalizeToUint8(cv, image) {
  const mm = cv.minMaxLoc(image);
  const out = new cv.Mat();
  if (mm.maxVal - mm.minVal < 1e-12) {
    out.create(image.rows, image.cols, cv.CV_8UC1);
    out.setTo(new cv.Scalar(0));
    return out;
  }
  const alpha = 255.0 / (mm.maxVal - mm.minVal);
  const beta = -mm.minVal * alpha;
  image.convertTo(out, cv.CV_8UC1, alpha, beta);
  return out;
}

// stage_a_specular_suppression_before_dog
export function buildSpecularMask(cv, srcBgr, sThresh = 0.15, vThresh = 0.75) {
  const hsv = new cv.Mat();
  cv.cvtColor(srcBgr, hsv, cv.COLOR_BGR2HSV);
  const chans = new cv.MatVector();
  cv.split(hsv, chans);
  const sChan = chans.get(1);
  const vChan = chans.get(2);

  const sMask = new cv.Mat();
  const vMask = new cv.Mat();
  cv.threshold(sChan, sMask, sThresh * 255, 255, cv.THRESH_BINARY_INV);
  cv.threshold(vChan, vMask, vThresh * 255, 255, cv.THRESH_BINARY);

  const specMask = new cv.Mat();
  cv.bitwise_and(sMask, vMask, specMask);

  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
  const dilated = new cv.Mat();
  cv.dilate(specMask, dilated, kernel, new cv.Point(-1, -1), 2);

  hsv.delete(); chans.delete(); sChan.delete(); vChan.delete();
  sMask.delete(); vMask.delete(); specMask.delete(); kernel.delete();
  return dilated;
}

// replace_masked_pixels_with_surround_average
export function suppressSpecular(cv, luminanceLog, specMask, sigmaS) {
  const surround = gaussianBlur(cv, luminanceLog, sigmaS);
  const cleaned = luminanceLog.clone();
  surround.copyTo(cleaned, specMask);
  surround.delete();
  return cleaned;
}

// stage_b_contour_area_filter_after_dog
export function filterByContourArea(cv, dog, dogThreshold = 5.0, minArea = 500) {
  const zero = new cv.Mat(dog.rows, dog.cols, dog.type(), new cv.Scalar(0));
  const absDog = new cv.Mat();
  cv.absdiff(dog, zero, absDog);
  zero.delete();

  const edgeBinaryF = new cv.Mat();
  cv.threshold(absDog, edgeBinaryF, dogThreshold, 255, cv.THRESH_BINARY);
  absDog.delete();
  const edgeBinary = new cv.Mat();
  edgeBinaryF.convertTo(edgeBinary, cv.CV_8UC1);
  edgeBinaryF.delete();

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(edgeBinary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  edgeBinary.delete(); hierarchy.delete();

  const filteredMask = cv.Mat.zeros(dog.rows, dog.cols, cv.CV_8UC1);
  const kept = new cv.MatVector();
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    if (cv.contourArea(c) > minArea) {
      kept.push_back(c);
    }
    c.delete();
  }
  contours.delete();
  if (kept.size() > 0) {
    cv.drawContours(filteredMask, kept, -1, new cv.Scalar(255, 255, 255, 255), -1);
  }
  kept.delete();
  return filteredMask;
}

// full_v2_pipeline: frameBgr (CV_8UC3) -> composite (CV_8UC3), caller owns and deletes the result
// outStats, if given, gets meanLuminance/isDim/dogThreshold written to it for the caller (e.g. tau tapering)
export function oplPipelineV2Core(cv, frameBgr, params = {}, outStats = null) {
  const {
    sigmaC = SIGMA_C_BASE,
    sigmaS = SIGMA_S_BASE,
    sThresh = 0.15,
    vThresh = 0.75,
    dogThreshold = 5.0,
    minContourArea = 500,
  } = params;

  // rgb_to_luminance
  const luminance = rgbToLuminance(cv, frameBgr);

  // scotopic_light_level_check
  const meanLum = meanLuminance(cv, luminance);
  const dim = isDimFrame(meanLum);

  // rod_convergence_pooling_before_log_transform
  const luminanceForLog = dim ? rodPooling(cv, luminance) : luminance;

  // log_transform
  const luminanceLog = logTransform(cv, luminanceForLog);
  if (dim) luminanceForLog.delete();
  luminance.delete();

  // specular_mask_and_suppression
  const specMask = buildSpecularMask(cv, frameBgr, sThresh, vThresh);
  const luminanceClean = suppressSpecular(cv, luminanceLog, specMask, sigmaS);
  luminanceLog.delete(); specMask.delete();

  // difference_of_gaussians_on_cleaned_luminance
  const dog = differenceOfGaussians(cv, luminanceClean, sigmaC, sigmaS);
  const onRaw = rectifyOn(cv, dog);
  const offRaw = rectifyOff(cv, dog);

  // noise_adaptive_dog_threshold_via_mad
  // a threshold can only reject noise it can measure - it can't recover an edge the camera itself never captured sharply (focus/compression/motion blur)
  const noiseSigma = estimateNoiseSigma(dog);
  const adaptiveDogThreshold = Math.max(dogThreshold, DOG_NOISE_MAD_MULTIPLIER * noiseSigma);

  // contour_area_filter
  const contourMask = filterByContourArea(cv, dog, adaptiveDogThreshold, minContourArea);
  dog.delete();

  if (outStats) {
    outStats.meanLuminance = meanLum;
    outStats.isDim = dim;
    outStats.dogThreshold = adaptiveDogThreshold;
  }
  const maskFloat = new cv.Mat();
  contourMask.convertTo(maskFloat, onRaw.type(), 1 / 255, 0);
  contourMask.delete();

  const onFiltered = new cv.Mat();
  cv.multiply(onRaw, maskFloat, onFiltered);
  const offFiltered = new cv.Mat();
  cv.multiply(offRaw, maskFloat, offFiltered);
  onRaw.delete(); offRaw.delete(); maskFloat.delete();

  // composite_grey_base_with_on_blue_off_red
  const base = new cv.Mat();
  luminanceClean.convertTo(base, cv.CV_8UC1);
  luminanceClean.delete();
  const compositeU8 = new cv.Mat();
  cv.cvtColor(base, compositeU8, cv.COLOR_GRAY2BGR);
  base.delete();
  const compositeF = new cv.Mat();
  compositeU8.convertTo(compositeF, cv.CV_32FC3);
  compositeU8.delete();

  const onNorm = normalizeToUint8(cv, onFiltered);
  const offNorm = normalizeToUint8(cv, offFiltered);
  onFiltered.delete(); offFiltered.delete();
  const onNormF = new cv.Mat(); onNorm.convertTo(onNormF, cv.CV_32F);
  const offNormF = new cv.Mat(); offNorm.convertTo(offNormF, cv.CV_32F);
  onNorm.delete(); offNorm.delete();

  const chans = new cv.MatVector();
  cv.split(compositeF, chans);
  compositeF.delete();
  const blue = chans.get(0), green = chans.get(1), red = chans.get(2);
  cv.add(blue, onNormF, blue);   // blue = ON
  cv.add(red, offNormF, red);    // red = OFF
  onNormF.delete(); offNormF.delete();

  const merged = new cv.MatVector();
  merged.push_back(blue); merged.push_back(green); merged.push_back(red);
  const mergedMat = new cv.Mat();
  cv.merge(merged, mergedMat);
  blue.delete(); green.delete(); red.delete(); chans.delete(); merged.delete();

  const result = new cv.Mat();
  mergedMat.convertTo(result, cv.CV_8UC3);
  mergedMat.delete();

  return result;
}

// exposure_to_alpha: leaky-integrator euler step, alpha = dt/tau clamped to [0,1]
export function exposureToAlpha(dtSeconds, tauSeconds) {
  if (tauSeconds <= 0) return 1.0;
  return clamp(dtSeconds / tauSeconds, 0.0, 1.0);
}

// gain_to_sigma_s
export function gainToSigmaS(gainNormalised, kGain = 0.5) {
  return SIGMA_S_BASE * (1.0 + kGain * clamp(gainNormalised, 0.0, 1.0));
}

// focus_to_sigma_c
export function focusToSigmaC(focusNormalised, kFocus = 0.5) {
  return SIGMA_C_BASE * (1.0 + kFocus * clamp(focusNormalised, 0.0, 1.0));
}

// camera_parameter_mapping_with_kuffler_ratio_clamp
export function computeSigmas(gainNormalised, focusNormalised, kGain = 0.5, kFocus = 0.5) {
  const sigmaC = focusToSigmaC(focusNormalised, kFocus);
  let sigmaS = gainToSigmaS(gainNormalised, kGain);
  sigmaS = clamp(sigmaS, 3.0 * sigmaC, 5.0 * sigmaC);
  return { sigmaC, sigmaS };
}

// y[n] = alpha*x[n] + (1-alpha)*y[n-1], euler discretisation of tau*dy/dt + y = x
export class TemporalBlender {
  constructor() {
    this.state = null; // cv.Mat CV_32FC3
  }

  // update: frameMat8u (CV_8UC3) -> blended CV_8UC3, caller owns and deletes the result
  update(cv, frameMat8u, alpha) {
    const frameF = new cv.Mat();
    frameMat8u.convertTo(frameF, cv.CV_32FC3);

    // camera can renegotiate resolution mid-stream; addWeighted requires
    // matching size/type, so a mismatched stale state must be dropped
    // rather than blended, or it throws and kills the frame loop
    if (this.state !== null &&
        (this.state.rows !== frameF.rows || this.state.cols !== frameF.cols || this.state.type() !== frameF.type())) {
      this.state.delete();
      this.state = null;
    }

    if (this.state === null) {
      this.state = frameF;
    } else {
      const blended = new cv.Mat();
      cv.addWeighted(frameF, alpha, this.state, 1.0 - alpha, 0, blended);
      this.state.delete();
      frameF.delete();
      this.state = blended;
    }

    const out = new cv.Mat();
    this.state.convertTo(out, cv.CV_8UC3);
    return out;
  }

  delete() {
    if (this.state !== null) {
      this.state.delete();
      this.state = null;
    }
  }
}

// rolling_average_over_last_n_frames
export class FPSMeter {
  constructor(windowSize = 30) {
    this.times = [];
    this.windowSize = windowSize;
  }

  tick(dt) {
    this.times.push(dt);
    if (this.times.length > this.windowSize) this.times.shift();
    const avg = this.times.reduce((a, b) => a + b, 0) / this.times.length;
    return avg > 0 ? 1.0 / avg : 0.0;
  }
}

// detect_track_capabilities: returns the capability object or null if unsupported
export function getTrackCapabilities(track) {
  if (!track || typeof track.getCapabilities !== 'function') return null;
  try {
    return track.getCapabilities();
  } catch (e) {
    return null;
  }
}

// map_normalised_0_1_into_a_capability_range_and_apply_via_hardware
export async function applyNormalisedConstraint(track, capabilityName, normalisedValue) {
  const caps = getTrackCapabilities(track);
  if (!caps || !(capabilityName in caps)) return false;
  const range = caps[capabilityName];
  if (typeof range.min !== 'number' || typeof range.max !== 'number') return false;
  const value = range.min + clamp(normalisedValue, 0, 1) * (range.max - range.min);
  try {
    await track.applyConstraints({ advanced: [{ [capabilityName]: value }] });
    return true;
  } catch (e) {
    return false;
  }
}
