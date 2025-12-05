"""
GIMBAL TRACKING SYSTEM BENCHMARK TOOL
======================================
Comprehensive testing and visualization for comparing original vs enhanced tracking systems.

FEATURES:
- Performance metrics collection (FPS, latency, jitter)
- Side-by-side comparison testing
- Publication-quality graphs and charts
- Statistical analysis and reporting
- Automated test scenarios

USAGE:
    python benchmark_tracking.py --test all --duration 60 --output results/
"""

import time
import numpy as np
import cv2
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import json
import argparse
from pathlib import Path


# ========================================
# METRICS COLLECTOR
# ========================================

class MetricsCollector:
    """
    Collects and analyzes performance metrics during tracking.
    """
    
    def __init__(self, name="System", max_samples=1000):
        self.name = name
        self.max_samples = max_samples
        
        # Time-series data
        self.timestamps = deque(maxlen=max_samples)
        self.fps_samples = deque(maxlen=max_samples)
        self.latency_samples = deque(maxlen=max_samples)
        self.pan_angles = deque(maxlen=max_samples)
        self.tilt_angles = deque(maxlen=max_samples)
        self.tracking_status = deque(maxlen=max_samples)
        self.confidence_scores = deque(maxlen=max_samples)
        
        # Performance counters
        self.frame_count = 0
        self.detection_count = 0
        self.lost_frames = 0
        self.start_time = None
        
        # Jitter calculation
        self.pan_velocity = deque(maxlen=100)
        self.tilt_velocity = deque(maxlen=100)
        
    def start_session(self):
        """Start a new metrics collection session."""
        self.start_time = time.time()
        self.frame_count = 0
        
    def record_frame(self, fps, latency, pan, tilt, tracking, confidence):
        """Record metrics for a single frame."""
        timestamp = time.time() - self.start_time if self.start_time else 0
        
        self.timestamps.append(timestamp)
        self.fps_samples.append(fps)
        self.latency_samples.append(latency)
        self.pan_angles.append(pan)
        self.tilt_angles.append(tilt)
        self.tracking_status.append(1 if tracking else 0)
        self.confidence_scores.append(confidence)
        
        self.frame_count += 1
        if tracking:
            self.detection_count += 1
        else:
            self.lost_frames += 1
        
        # Calculate velocity for jitter analysis
        if len(self.pan_angles) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            if dt > 0:
                pan_vel = abs(self.pan_angles[-1] - self.pan_angles[-2]) / dt
                tilt_vel = abs(self.tilt_angles[-1] - self.tilt_angles[-2]) / dt
                self.pan_velocity.append(pan_vel)
                self.tilt_velocity.append(tilt_vel)
    
    def get_statistics(self):
        """Calculate comprehensive statistics."""
        if not self.fps_samples:
            return {}
        
        stats = {
            'name': self.name,
            'duration': self.timestamps[-1] if self.timestamps else 0,
            'total_frames': self.frame_count,
            'detected_frames': self.detection_count,
            'lost_frames': self.lost_frames,
            'tracking_rate': self.detection_count / max(self.frame_count, 1),
            
            # FPS statistics
            'fps_mean': np.mean(self.fps_samples),
            'fps_std': np.std(self.fps_samples),
            'fps_min': np.min(self.fps_samples),
            'fps_max': np.max(self.fps_samples),
            'fps_p50': np.percentile(self.fps_samples, 50),
            'fps_p95': np.percentile(self.fps_samples, 95),
            
            # Latency statistics (ms)
            'latency_mean': np.mean(self.latency_samples),
            'latency_std': np.std(self.latency_samples),
            'latency_min': np.min(self.latency_samples),
            'latency_max': np.max(self.latency_samples),
            'latency_p50': np.percentile(self.latency_samples, 50),
            'latency_p95': np.percentile(self.latency_samples, 95),
            
            # Jitter (RMS of angular velocity)
            'pan_jitter_rms': np.sqrt(np.mean(np.array(self.pan_velocity)**2)) if self.pan_velocity else 0,
            'tilt_jitter_rms': np.sqrt(np.mean(np.array(self.tilt_velocity)**2)) if self.tilt_velocity else 0,
            
            # Confidence statistics
            'confidence_mean': np.mean([c for c in self.confidence_scores if c > 0]) if any(self.confidence_scores) else 0,
            'confidence_std': np.std([c for c in self.confidence_scores if c > 0]) if any(self.confidence_scores) else 0,
        }
        
        # Overall jitter metric
        stats['overall_jitter_rms'] = np.sqrt(stats['pan_jitter_rms']**2 + stats['tilt_jitter_rms']**2)
        
        return stats
    
    def export_timeseries(self):
        """Export time-series data for plotting."""
        return {
            'timestamps': list(self.timestamps),
            'fps': list(self.fps_samples),
            'latency': list(self.latency_samples),
            'pan': list(self.pan_angles),
            'tilt': list(self.tilt_angles),
            'tracking': list(self.tracking_status),
            'confidence': list(self.confidence_scores),
        }


# ========================================
# VISUALIZATION GENERATOR
# ========================================

class BenchmarkVisualizer:
    """
    Generates publication-quality comparison charts.
    """
    
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'original': '#E74C3C',
            'enhanced': '#2ECC71',
            'neutral': '#3498DB'
        }
    
    def plot_fps_comparison(self, original_data, enhanced_data):
        """
        Plot FPS over time for both systems.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time series
        ax1.plot(original_data['timestamps'], original_data['fps'], 
                 color=self.colors['original'], linewidth=1.5, alpha=0.7, label='Original')
        ax1.plot(enhanced_data['timestamps'], enhanced_data['fps'], 
                 color=self.colors['enhanced'], linewidth=1.5, alpha=0.7, label='Enhanced')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('FPS', fontsize=12)
        ax1.set_title('Frame Rate Comparison Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.hist(original_data['fps'], bins=50, alpha=0.6, 
                 color=self.colors['original'], label='Original', density=True)
        ax2.hist(enhanced_data['fps'], bins=50, alpha=0.6, 
                 color=self.colors['enhanced'], label='Enhanced', density=True)
        ax2.set_xlabel('FPS', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('FPS Distribution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fps_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved: fps_comparison.png")
    
    def plot_latency_comparison(self, original_data, enhanced_data):
        """
        Plot latency comparison.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time series
        ax1.plot(original_data['timestamps'], original_data['latency'], 
                 color=self.colors['original'], linewidth=1.5, alpha=0.7, label='Original')
        ax1.plot(enhanced_data['timestamps'], enhanced_data['latency'], 
                 color=self.colors['enhanced'], linewidth=1.5, alpha=0.7, label='Enhanced')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('Detection-to-Servo Latency', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [original_data['latency'], enhanced_data['latency']]
        bp = ax2.boxplot(data_to_plot, labels=['Original', 'Enhanced'],
                         patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor(self.colors['original'])
        bp['boxes'][1].set_facecolor(self.colors['enhanced'])
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.set_title('Latency Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved: latency_comparison.png")
    
    def plot_tracking_quality(self, original_data, enhanced_data):
        """
        Plot tracking quality metrics (jitter, smoothness).
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Pan angle over time
        ax1.plot(original_data['timestamps'], original_data['pan'], 
                 color=self.colors['original'], linewidth=1.5, alpha=0.7, label='Original')
        ax1.plot(enhanced_data['timestamps'], enhanced_data['pan'], 
                 color=self.colors['enhanced'], linewidth=1.5, alpha=0.7, label='Enhanced')
        ax1.set_xlabel('Time (seconds)', fontsize=11)
        ax1.set_ylabel('Pan Angle (°)', fontsize=11)
        ax1.set_title('Pan Servo Angle', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Tilt angle over time
        ax2.plot(original_data['timestamps'], original_data['tilt'], 
                 color=self.colors['original'], linewidth=1.5, alpha=0.7, label='Original')
        ax2.plot(enhanced_data['timestamps'], enhanced_data['tilt'], 
                 color=self.colors['enhanced'], linewidth=1.5, alpha=0.7, label='Enhanced')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Tilt Angle (°)', fontsize=11)
        ax2.set_title('Tilt Servo Angle', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Tracking status
        ax3.fill_between(original_data['timestamps'], 0, original_data['tracking'], 
                         color=self.colors['original'], alpha=0.5, label='Original')
        ax3.fill_between(enhanced_data['timestamps'], 0, enhanced_data['tracking'], 
                         color=self.colors['enhanced'], alpha=0.5, label='Enhanced')
        ax3.set_xlabel('Time (seconds)', fontsize=11)
        ax3.set_ylabel('Tracking Status', fontsize=11)
        ax3.set_title('Target Lock Status (1=Tracking, 0=Lost)', fontsize=12, fontweight='bold')
        ax3.set_ylim(-0.1, 1.1)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Confidence over time
        orig_conf = [c if c > 0 else np.nan for c in original_data['confidence']]
        enh_conf = [c if c > 0 else np.nan for c in enhanced_data['confidence']]
        ax4.plot(original_data['timestamps'], orig_conf, 
                 color=self.colors['original'], linewidth=1.5, alpha=0.7, label='Original')
        ax4.plot(enhanced_data['timestamps'], enh_conf, 
                 color=self.colors['enhanced'], linewidth=1.5, alpha=0.7, label='Enhanced')
        ax4.set_xlabel('Time (seconds)', fontsize=11)
        ax4.set_ylabel('Detection Confidence', fontsize=11)
        ax4.set_title('YOLO Detection Confidence', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tracking_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved: tracking_quality.png")
    
    def plot_summary_bars(self, original_stats, enhanced_stats):
        """
        Create bar chart comparing key metrics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        metrics = [
            ('fps_mean', 'Average FPS', 'FPS', True),
            ('latency_mean', 'Average Latency', 'ms', False),
            ('overall_jitter_rms', 'Servo Jitter (RMS)', '°/s', False),
            ('tracking_rate', 'Tracking Success Rate', '%', True),
            ('fps_p95', '95th Percentile FPS', 'FPS', True),
            ('latency_p95', '95th Percentile Latency', 'ms', False),
        ]
        
        for idx, (metric, title, unit, higher_better) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            orig_val = original_stats.get(metric, 0)
            enh_val = enhanced_stats.get(metric, 0)
            
            # Convert tracking rate to percentage
            if metric == 'tracking_rate':
                orig_val *= 100
                enh_val *= 100
            
            bars = ax.bar(['Original', 'Enhanced'], [orig_val, enh_val],
                          color=[self.colors['original'], self.colors['enhanced']], 
                          alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Calculate improvement
            if orig_val != 0:
                if higher_better:
                    improvement = ((enh_val - orig_val) / orig_val) * 100
                else:
                    improvement = ((orig_val - enh_val) / orig_val) * 100
                
                color = 'green' if improvement > 0 else 'red'
                ax.text(0.5, 0.95, f'{improvement:+.1f}%', 
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=12, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_ylabel(unit, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved: summary_comparison.png")
    
    def plot_performance_radar(self, original_stats, enhanced_stats):
        """
        Create radar chart for multi-dimensional comparison.
        """
        # Normalize metrics to 0-100 scale
        metrics = {
            'FPS': ('fps_mean', 60, True),  # (key, max_value, higher_better)
            'Latency': ('latency_mean', 150, False),
            'Jitter': ('overall_jitter_rms', 5, False),
            'Tracking\nRate': ('tracking_rate', 1, True),
            'Stability': ('fps_std', 15, False),
        }
        
        categories = list(metrics.keys())
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate normalized values
        original_values = []
        enhanced_values = []
        
        for category, (key, max_val, higher_better) in metrics.items():
            orig = original_stats.get(key, 0)
            enh = enhanced_stats.get(key, 0)
            
            # Normalize to 0-100
            if higher_better:
                orig_norm = (orig / max_val) * 100
                enh_norm = (enh / max_val) * 100
            else:
                orig_norm = ((max_val - orig) / max_val) * 100
                enh_norm = ((max_val - enh) / max_val) * 100
            
            # Clamp to 0-100
            orig_norm = max(0, min(100, orig_norm))
            enh_norm = max(0, min(100, enh_norm))
            
            original_values.append(orig_norm)
            enhanced_values.append(enh_norm)
        
        original_values += original_values[:1]
        enhanced_values += enhanced_values[:1]
        
        # Plot
        ax.plot(angles, original_values, 'o-', linewidth=2, 
                color=self.colors['original'], label='Original', markersize=8)
        ax.fill(angles, original_values, alpha=0.25, color=self.colors['original'])
        
        ax.plot(angles, enhanced_values, 'o-', linewidth=2, 
                color=self.colors['enhanced'], label='Enhanced', markersize=8)
        ax.fill(angles, enhanced_values, alpha=0.25, color=self.colors['enhanced'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.set_title('Performance Comparison Radar\n(Higher = Better)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved: radar_comparison.png")
    
    def generate_report(self, original_stats, enhanced_stats):
        """
        Generate a text report with detailed statistics.
        """
        report_path = self.output_dir / 'benchmark_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GIMBAL TRACKING SYSTEM BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            # Calculate improvements
            improvements = {}
            for key in original_stats:
                if key in enhanced_stats and isinstance(original_stats[key], (int, float)):
                    orig = original_stats[key]
                    enh = enhanced_stats[key]
                    if orig != 0:
                        improvements[key] = ((enh - orig) / orig) * 100
            
            # Format statistics
            stats_format = [
                ("Test Duration", "duration", "s", 1),
                ("Total Frames", "total_frames", "", 0),
                ("", "", "", 0),
                ("Average FPS", "fps_mean", "FPS", 1),
                ("FPS Std Dev", "fps_std", "FPS", 2),
                ("Min FPS", "fps_min", "FPS", 1),
                ("Max FPS", "fps_max", "FPS", 1),
                ("50th Percentile FPS", "fps_p50", "FPS", 1),
                ("95th Percentile FPS", "fps_p95", "FPS", 1),
                ("", "", "", 0),
                ("Average Latency", "latency_mean", "ms", 1),
                ("Latency Std Dev", "latency_std", "ms", 2),
                ("Min Latency", "latency_min", "ms", 1),
                ("Max Latency", "latency_max", "ms", 1),
                ("95th Percentile Latency", "latency_p95", "ms", 1),
                ("", "", "", 0),
                ("Pan Jitter (RMS)", "pan_jitter_rms", "°/s", 2),
                ("Tilt Jitter (RMS)", "tilt_jitter_rms", "°/s", 2),
                ("Overall Jitter (RMS)", "overall_jitter_rms", "°/s", 2),
                ("", "", "", 0),
                ("Tracking Success Rate", "tracking_rate", "%", 1),
                ("Average Confidence", "confidence_mean", "", 3),
            ]
            
            f.write(f"{'Metric':<30} {'Original':<15} {'Enhanced':<15} {'Change':<12}\n")
            f.write("-" * 80 + "\n")
            
            for label, key, unit, decimals in stats_format:
                if not label:  # Empty line
                    f.write("\n")
                    continue
                
                orig_val = original_stats.get(key, 0)
                enh_val = enhanced_stats.get(key, 0)
                
                # Special formatting for tracking rate
                if key == "tracking_rate":
                    orig_val *= 100
                    enh_val *= 100
                
                if decimals == 0:
                    orig_str = f"{orig_val:.0f} {unit}".strip()
                    enh_str = f"{enh_val:.0f} {unit}".strip()
                else:
                    orig_str = f"{orig_val:.{decimals}f} {unit}".strip()
                    enh_str = f"{enh_val:.{decimals}f} {unit}".strip()
                
                change = improvements.get(key, 0)
                change_str = f"{change:+.1f}%" if key in improvements else "N/A"
                
                f.write(f"{label:<30} {orig_str:<15} {enh_str:<15} {change_str:<12}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("KEY IMPROVEMENTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Highlight major improvements
            key_improvements = [
                ("FPS Improvement", improvements.get('fps_mean', 0)),
                ("Latency Reduction", -improvements.get('latency_mean', 0)),
                ("Jitter Reduction", -improvements.get('overall_jitter_rms', 0)),
                ("Tracking Rate Improvement", improvements.get('tracking_rate', 0)),
            ]
            
            for label, value in key_improvements:
                if value > 0:
                    f.write(f"✓ {label}: {value:.1f}%\n")
                else:
                    f.write(f"✗ {label}: {value:.1f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"[INFO] Saved: benchmark_report.txt")
        
        # Also save JSON for programmatic access
        json_data = {
            'original': original_stats,
            'enhanced': enhanced_stats,
            'improvements': improvements,
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = self.output_dir / 'benchmark_data.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"[INFO] Saved: benchmark_data.json")


# ========================================
# SIMULATED BENCHMARK RUNNER
# ========================================

class SimulatedBenchmark:
    """
    Simulates tracking system performance for testing visualization.
    """
    
    @staticmethod
    def generate_original_system_data(duration=30, fps_target=15):
        """
        Simulate original system with realistic performance characteristics.
        """
        collector = MetricsCollector("Original System")
        collector.start_session()
        
        dt = 1.0 / fps_target
        t = 0
        
        while t < duration:
            # Simulate FPS variability (8-20 FPS range)
            fps = np.random.normal(15, 3)
            fps = max(8, min(20, fps))
            
            # Simulate latency (100-150ms range)
            latency = np.random.normal(125, 15)
            latency = max(100, min(150, latency))
            
            # Simulate servo angles with jitter
            pan = 90 + 20 * np.sin(t * 0.5) + np.random.normal(0, 2)
            tilt = 90 + 15 * np.cos(t * 0.3) + np.random.normal(0, 2)
            
            # Simulate occasional tracking loss
            tracking = np.random.random() > 0.1
            confidence = np.random.uniform(0.7, 0.95) if tracking else 0
            
            collector.record_frame(fps, latency, pan, tilt, tracking, confidence)
            
            t += dt
            time.sleep(0.001)  # Minimal delay for simulation
        
        return collector
    
    @staticmethod
    def generate_enhanced_system_data(duration=30, fps_target=60):
        """
        Simulate enhanced system with improved performance.
        """
        collector = MetricsCollector("Enhanced System")
        collector.start_session()
        
        dt = 1.0 / fps_target
        t = 0
        
        while t < duration:
            # Simulate FPS variability (55-65 FPS range)
            fps = np.random.normal(60, 2)
            fps = max(55, min(65, fps))
            
            # Simulate latency (30-50ms range)
            latency = np.random.normal(40, 5)
            latency = max(30, min(50, latency))
            
            # Simulate servo angles with less jitter (Kalman smoothing)
            pan = 90 + 20 * np.sin(t * 0.5) + np.random.normal(0, 0.5)
            tilt = 90 + 15 * np.cos(t * 0.3) + np.random.normal(0, 0.5)
            
            # Better tracking rate
            tracking = np.random.random() > 0.02
            confidence = np.random.uniform(0.75, 0.98) if tracking else 0
            
            collector.record_frame(fps, latency, pan, tilt, tracking, confidence)
            
            t += dt
            time.sleep(0.001)
        
        return collector


# ========================================
# MAIN BENCHMARK EXECUTION
# ========================================

def run_benchmark(args):
    """
    Run complete benchmark suite and generate all visualizations.
    """
    print("=" * 80)
    print("GIMBAL TRACKING SYSTEM BENCHMARK")
    print("=" * 80)
    print(f"Duration: {args.duration} seconds")
    print(f"Output directory: {args.output}")
    print()
    
    # Generate or collect data
    print("[INFO] Collecting performance data...")
    
    if args.mode == 'simulate':
        print("[INFO] Running in SIMULATION mode")
        original_collector = SimulatedBenchmark.generate_original_system_data(args.duration)
        enhanced_collector = SimulatedBenchmark.generate_enhanced_system_data(args.duration)
    else:
        print("[ERROR] Live benchmark not yet implemented")
        print("[INFO] Use --mode simulate for now")
        return
    
    # Get statistics
    print("[INFO] Calculating statistics...")
    original_stats = original_collector.get_statistics()
    enhanced_stats = enhanced_collector.get_statistics()
    
    # Export time-series data
    original_data = original_collector.export_timeseries()
    enhanced_data = enhanced_collector.export_timeseries()
    
    # Generate visualizations
    print("[INFO] Generating visualizations...")
    viz = BenchmarkVisualizer(args.output)
    
    viz.plot_fps_comparison(original_data, enhanced_data)
    viz.plot_latency_comparison(original_data, enhanced_data)
    viz.plot_tracking_quality(original_data, enhanced_data)
    viz.plot_summary_bars(original_stats, enhanced_stats)
    viz.plot_performance_radar(original_stats, enhanced_stats)
    viz.generate_report(original_stats, enhanced_stats)
    
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output}/")
    print()
    print("Generated files:")
    print("  - fps_comparison.png")
    print("  - latency_comparison.png")
    print("  - tracking_quality.png")
    print("  - summary_comparison.png")
    print("  - radar_comparison.png")
    print("  - benchmark_report.txt")
    print("  - benchmark_data.json")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark gimbal tracking systems and generate comparison visualizations'
    )
    parser.add_argument('--mode', choices=['simulate', 'live'], default='simulate',
                        help='Benchmark mode: simulate or live testing')
    parser.add_argument('--duration', type=int, default=30,
                        help='Test duration in seconds (default: 30)')
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory for results (default: benchmark_results)')
    
    args = parser.parse_args()
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
