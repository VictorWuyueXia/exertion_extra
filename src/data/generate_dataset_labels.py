import argparse
import csv
import os
import re
from typing import Dict, List, Tuple, Optional


def read_exertion_mapping(general_info_csv_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
	"""Read general_information.csv and build two mappings:

	- exact_map: full session name (up to specific clip id) -> exertion
	- root_map: session root (prefix up to and including '_clip_') -> exertion

	The CSV is expected to have a header with the first column named 'Session Name' and
	second column 'Exertion'. Additional columns are ignored.
	"""
	exact_map: Dict[str, int] = {}
	root_map: Dict[str, int] = {}
	with open(general_info_csv_path, "r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		# Normalize header names to handle potential BOM or spacing
		field_map = {name.strip(): name for name in reader.fieldnames or []}
		session_key = field_map.get("Session Name")
		exertion_key = field_map.get("Exertion")
		if not session_key or not exertion_key:
			raise ValueError(
				f"Missing required columns in {general_info_csv_path}. Found: {reader.fieldnames}"
			)
		for row in reader:
			session_name = (row.get(session_key) or "").strip()
			exertion_raw = (row.get(exertion_key) or "").strip()
			if not session_name:
				continue
			# Best-effort cast to int; skip if not numeric
			try:
				exertion_val = int(exertion_raw)
			except ValueError:
				continue
			exact_map[session_name] = exertion_val
			root_key = extract_session_root(session_name)
			if root_key:
				# In case of conflicts, keep the first seen value if they are consistent; if not, overwrite but warn
				prev = root_map.get(root_key)
				if prev is None or prev == exertion_val:
					root_map[root_key] = exertion_val
				else:
					# Prefer the latest value but could log a warning
					root_map[root_key] = exertion_val
	return exact_map, root_map


_SUFFIX_NORMALIZATION_PATTERNS: List[re.Pattern] = [
	# Common suffixes seen in data filenames
	re.compile(r"(_first)$"),
	re.compile(r"(_last15s)$"),
]


def normalize_session_name(basename_without_ext: str) -> str:
	"""Normalize an audio basename to match keys in general_information.

	- Remove known trailing suffixes like '_first' or '_last15s'
	- Ensure it ends at the first clip id token: '_clip_<num>' (ignore anything that follows)
	"""
	name = basename_without_ext
	# Trim common suffixes
	for pat in _SUFFIX_NORMALIZATION_PATTERNS:
		name = pat.sub("", name)
	# Truncate to '_clip_<num>' if any extra tokens follow
	clip_match = re.search(r"_clip_(\d+)", name)
	if clip_match:
		end = clip_match.end()
		name = name[:end]
	return name


def extract_session_root(name: str) -> Optional[str]:
	"""Return prefix up to and including '_clip_' or None if pattern not found."""
	m = re.search(r"_clip_", name)
	if not m:
		return None
	return name[: m.end()]


def extract_task_from_filename(filename: str) -> int:
	"""Extract the first integer that appears right after 'clip_' in filename.

	Examples:
	- '..._clip_1.wav' -> 1
	- '..._clip_1_first.wav' -> 1
	- '..._clip_12_foo.wav' -> 12
	"""
	match = re.search(r"clip_(\d+)", filename)
	if not match:
		raise ValueError(f"Cannot extract task id from filename: {filename}")
	return int(match.group(1))


def gather_audio_files(audio_dir: str) -> List[str]:
	return sorted([
		f for f in os.listdir(audio_dir)
		if os.path.isfile(os.path.join(audio_dir, f)) and f.lower().endswith(".wav")
	])


def generate_labels_for_split(audio_dir: str, exact_map: Dict[str, int], root_map: Dict[str, int]) -> List[Tuple[str, int, int]]:
	"""Generate (filename, exertion, task) rows for a given split audio directory."""
	rows: List[Tuple[str, int, int]] = []
	for fname in gather_audio_files(audio_dir):
		basename_no_ext = os.path.splitext(fname)[0]
		normalized_key = normalize_session_name(basename_no_ext)
		exertion = exact_map.get(normalized_key)
		if exertion is None:
			# Fallback 1: try root mapping (prefix up to '_clip_')
			root_key = extract_session_root(normalized_key)
			if root_key is not None:
				exertion = root_map.get(root_key)
		if exertion is None:
			# Last resort: try original basename (no normalization)
			exertion = exact_map.get(basename_no_ext)
		if exertion is None:
			# Skip if we cannot find a mapping; alternatively, raise to fail-fast
			raise KeyError(
				f"Exertion not found for audio '{fname}'. Tried keys: "
				f"'{normalized_key}', root='{extract_session_root(normalized_key)}', raw='{basename_no_ext}'"
			)
		task_id = extract_task_from_filename(fname)
		rows.append((fname, exertion, task_id))
	return rows


def write_labels_csv(output_csv_path: str, rows: List[Tuple[str, int, int]]) -> None:
	with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["filename", "exertion", "task"])
		for fname, exertion, task_id in rows:
			writer.writerow([fname, exertion, task_id])


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate labels.csv for new_split train/val/test datasets.")
	parser.add_argument(
		"--data-root",
		type=str,
		default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data"),
		help="Path to the Data directory that contains general_information.csv and new_split_* folders.",
	)
	parser.add_argument(
		"--splits",
		nargs="*",
		default=["new_split_train", "new_split_val", "new_split_test"],
		help="List of split folder names under Data to process.",
	)
	args = parser.parse_args()

	general_info_csv = os.path.join(args.data_root, "general_information.csv")
	if not os.path.isfile(general_info_csv):
		raise FileNotFoundError(f"general_information.csv not found at: {general_info_csv}")

	exact_map, root_map = read_exertion_mapping(general_info_csv)

	for split_name in args.splits:
		split_dir = os.path.join(args.data_root, split_name)
		audio_dir = os.path.join(split_dir, "audio")
		if not os.path.isdir(audio_dir):
			raise FileNotFoundError(f"Audio directory not found for split '{split_name}': {audio_dir}")
		rows = generate_labels_for_split(audio_dir, exact_map, root_map)
		output_csv = os.path.join(split_dir, "labels.csv")
		write_labels_csv(output_csv, rows)
		print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
	main()


