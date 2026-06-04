"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface GlassPanelProps {
  title: string;
  /** Retained for API compatibility; the Swiss layout renders a numeric index instead. */
  icon?: string;
  /** Two-digit section index shown flush-left in the panel header. */
  index?: string;
  children: ReactNode;
  className?: string;
  delay?: number;
}

export default function GlassPanel({
  title,
  index,
  children,
  className = "",
  delay = 0,
}: GlassPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
      className={`glass-panel group ${className}`}
    >
      <div className="panel-header">
        <div className="flex items-baseline">
          {index && <span className="panel-index">{index}</span>}
          <h2 className="panel-title">{title}</h2>
        </div>
        <div className="panel-dot-group">
          <span className="panel-dot panel-dot--green" />
          <span className="panel-dot panel-dot--yellow" />
          <span className="panel-dot panel-dot--red" />
        </div>
      </div>
      <div className="panel-content">{children}</div>
    </motion.div>
  );
}
