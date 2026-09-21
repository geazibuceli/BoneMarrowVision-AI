export type Prediction = {
  id: number;
  model: { id: number; name: string; version: string; architecture: string } | null;
  predicted_class: string;
  confidence: number;
  top_predictions_json: { class: string; probability: number }[];
  gradcam_image_path: string | null;
  low_confidence: boolean;
  created_at: string;
  disclaimer: string;
};

export type User = { id: number; email: string; full_name: string; role: string };
export type Model = { id: number; name: string; version: string; architecture: string; metrics_json: Record<string, unknown>; checkpoint_sha256: string | null; is_active: boolean };
export type Experiment = { id: number; name: string; dataset_version: string; validation_metrics_json: Record<string, number>; test_metrics_json: Record<string, number>; created_at: string };
