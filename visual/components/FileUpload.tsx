"use client";

import { useCallback } from "react";
import { motion } from "framer-motion";

interface FileUploadProps {
  onData: (data: unknown) => void;
}

export default function FileUpload({ onData }: FileUploadProps) {
  const handleFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target?.result as string);
          onData(json);
        } catch {
          alert("Failed to parse JSON file");
        }
      };
      reader.readAsText(file);
    },
    [onData]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <motion.label
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="file-upload-zone"
    >
      <input
        type="file"
        accept=".json"
        onChange={handleChange}
        className="hidden"
      />
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        className="opacity-50"
      >
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <span className="text-xs opacity-60">
        Drop <code>out.json</code> or click to upload
      </span>
    </motion.label>
  );
}
