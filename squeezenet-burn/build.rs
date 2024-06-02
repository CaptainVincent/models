use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

const LABEL_SOURCE_FILE: &str = "src/model/label.txt";
const LABEL_DEST_FILE: &str = "src/model/label.rs";
const GENERATED_MODEL_FILE: &str = "squeezenet1.rs";
const GENERATED_MODEL_WEIGHTS_FILE: &str = "squeezenet1.mpk";
const MODEL_DEST_FOLDER: &str = "src/model";
const INPUT_ONNX_FILE: &str = "src/model/squeezenet1.onnx";
const OUT_DIR: &str = "model/";

fn main() {
    // Re-run the build script if model files change.
    println!("cargo:rerun-if-changed=src/model");

    // Make sure either weights_file or weights_embedded is enabled.
    if cfg!(feature = "weights_file") && cfg!(feature = "weights_embedded") {
        panic!("Only one of the features weights_file and weights_embedded can be enabled");
    }

    // Make sure at least one of weights_file or weights_embedded is enabled.
    if !cfg!(feature = "weights_file") && !cfg!(feature = "weights_embedded") {
        panic!("One of the features weights_file and weights_embedded must be enabled");
    }

    #[cfg(feature = "rebuild")]
    {
        use burn_import::burn::graph::RecordType;
        use burn_import::onnx::ModelGen;

        // Check if the weights are embedded.
        let (record_type, embed_states) = if cfg!(feature = "weights_embedded") {
            (RecordType::Bincode, true)
        } else {
            (RecordType::NamedMpk, false)
        };

        // Check if half precision is enabled.
        let half_precision = cfg!(feature = "half_precision");

        // Generate the model code from the ONNX file.
        ModelGen::new()
            .input(INPUT_ONNX_FILE)
            .out_dir(OUT_DIR)
            .record_type(record_type)
            .embed_states(embed_states)
            .half_precision(half_precision)
            .run_from_script();

        // Refresh model code
        extract_model_code();

        // Generate the labels from the synset.txt file.
        generate_labels_from_txt_file().unwrap();
    }

    // Copy the weights next to the executable.
    if cfg!(feature = "weights_file") {
        extract_model_wweights();
    }
}

/// Read labels from synset.txt and store them in a vector of strings in a Rust file.
fn generate_labels_from_txt_file() -> std::io::Result<()> {
    let dest_path = Path::new(LABEL_DEST_FILE);
    let mut f = File::create(dest_path)?;

    let file = File::open(LABEL_SOURCE_FILE)?;
    let reader = BufReader::new(file);

    writeln!(f, "pub static LABELS: &[&str] = &[")?;
    for line in reader.lines() {
        writeln!(f, "    \"{}\",", line.unwrap())?;
    }
    writeln!(f, "];")?;

    Ok(())
}

/// Cache the model file
fn extract_model_code() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not defined");
    let dest_folder = Path::new(&MODEL_DEST_FOLDER);

    let source_path = Path::new(&out_dir).join("model").join(GENERATED_MODEL_FILE);
    let destination_path = dest_folder.join(GENERATED_MODEL_FILE);
    fs::copy(source_path, destination_path).expect("Failed to copy generated modle file");
}

/// Cache the weights file first then copy it next to the executable.
fn extract_model_wweights() {
    #[cfg(feature = "rebuild")]
    {
        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not defined");
        let source_path = Path::new(&out_dir)
            .join("model")
            .join(GENERATED_MODEL_WEIGHTS_FILE);
        let dest_folder = Path::new(&MODEL_DEST_FOLDER);
        let destination_path = dest_folder.join(GENERATED_MODEL_WEIGHTS_FILE);
        fs::copy(source_path, destination_path)
            .expect("Failed to copy generated modle weight file");
    }

    // Determine the profile (debug or release) to set the appropriate destination directory.
    let profile = env::var("PROFILE").expect("PROFILE not defined");
    let target_dir = format!("target/{}", profile);
    let destination_folder = Path::new(&target_dir).join("examples");
    if destination_folder.exists() && destination_folder.is_dir() {
        let destination_path = destination_folder.join(GENERATED_MODEL_WEIGHTS_FILE);
        let source_path = Path::new(&MODEL_DEST_FOLDER).join(GENERATED_MODEL_WEIGHTS_FILE);
        fs::copy(source_path, destination_path)
            .expect("Failed to copy generated model weight file");
    }
}
