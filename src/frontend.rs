//! Bindings for the audio "frontend" library for feature generation
//!
//! This API is in the 'experimental' directory in tensorflow. Therefore
//! these bindings should also be considered to be experimental.
//!
//! See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/microfrontend/lib

#![allow(non_snake_case)]

use core::slice;

use crate::bindings;

cpp! {{
    #include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
    #include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}}

/// Bindings for the audio "frontend" library for feature generation
pub struct Frontend(bindings::FrontendState);

// Frontend allocates memory on the heap, therefore the raw pointers that
// in contains are Send
unsafe impl Send for Frontend {}

impl Frontend {
    /// Create new frontend state
    pub fn new() -> Result<Self, ()> {
        let mut config: bindings::FrontendConfig = Default::default();
        let mut state: bindings::FrontendState = Default::default();

        let kFeatureSliceDurationMs = 30;
        let kFeatureSliceStrideMs = 20;
        let kFeatureSliceSize = 40;
        let kAudioSampleFrequency: u32 = 16_000;

        config.window.size_ms = kFeatureSliceDurationMs;
        config.window.step_size_ms = kFeatureSliceStrideMs;
        config.noise_reduction.smoothing_bits = 10;
        config.filterbank.num_channels = kFeatureSliceSize;
        config.filterbank.lower_band_limit = 125.0;
        config.filterbank.upper_band_limit = 7500.0;
        config.noise_reduction.smoothing_bits = 10;
        config.noise_reduction.even_smoothing = 0.025;
        config.noise_reduction.odd_smoothing = 0.06;
        config.noise_reduction.min_signal_remaining = 0.05;
        config.pcan_gain_control.enable_pcan = 1;
        config.pcan_gain_control.strength = 0.95;
        config.pcan_gain_control.offset = 80.0;
        config.pcan_gain_control.gain_bits = 21;
        config.log_scale.enable_log = 1;
        config.log_scale.scale_shift = 6;

        let state_ref = &mut state;

        let result = cpp! (unsafe [
            config as "FrontendConfig",
            state_ref as "FrontendState*",
            kAudioSampleFrequency as "int"
        ] -> u32 as "int" {
            return FrontendPopulateState(&config,
                                         state_ref,
                                         kAudioSampleFrequency);
        });

        if result == 1 {
            Ok(Self(state))
        } else {
            Err(())
        }
    }

    /// Generates micro_features objects
    ///
    /// Returns num_samples_read
    pub fn generate_micro_features(
        &mut self,
        input: &[i16],
        output: &mut [u16],
    ) -> usize {
        let micro_features_state_ref = &mut self.0;

        let len = input.len();
        let input = input.as_ptr();

        let mut num_samples_read = 0usize;
        let num_samples_read_ref = &mut num_samples_read;

        let frontend_output = cpp! (unsafe [
            micro_features_state_ref as "FrontendState*",
            input as "int16_t*",
            len as "size_t",
            num_samples_read_ref as "size_t*"
        ] -> bindings::FrontendOutput as "FrontendOutput" {
            return FrontendProcessSamples(
                micro_features_state_ref,
                input,
                len,
                num_samples_read_ref
            );
        });

        let frontend_output_slice = unsafe {
            slice::from_raw_parts(frontend_output.values, frontend_output.size)
        };

        assert_eq!(frontend_output_slice.len(), output.len());

        // Copy out the raw data, this still needs to be scaled
        output.clone_from_slice(frontend_output_slice);

        num_samples_read
    }
}
