"""
FUK Logger - Centralized logging with timestamps and categories
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import time
import traceback


class FukLogger:
    """Centralized logging with timestamps and categories"""
    
    COLORS = {
        'header': '\033[95m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    
    # Track timing for operations
    _timers: Dict[str, float] = {}
    
    @staticmethod
    def timestamp():
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    @classmethod
    def header(cls, title: str, char: str = "="):
        width = 70
        print(f"\n{cls.COLORS['cyan']}{char * width}")
        print(f"  {title}")
        print(f"{char * width}{cls.COLORS['end']}\n")
    
    @classmethod
    def section(cls, title: str):
        print(f"\n{cls.COLORS['blue']}--- {title} ---{cls.COLORS['end']}")
    
    @classmethod
    def info(cls, category: str, message: str):
        print(f"{cls.COLORS['cyan']}[{cls.timestamp()}]{cls.COLORS['end']} [{category}] {message}")
    
    @classmethod
    def success(cls, category: str, message: str):
        print(f"{cls.COLORS['green']}[{cls.timestamp()}] ✓ [{category}] {message}{cls.COLORS['end']}")
    
    @classmethod
    def warning(cls, category: str, message: str):
        print(f"{cls.COLORS['yellow']}[{cls.timestamp()}] ⚠ [{category}] {message}{cls.COLORS['end']}")
    
    @classmethod
    def error(cls, category: str, message: str):
        print(f"{cls.COLORS['red']}[{cls.timestamp()}] ✗ [{category}] {message}{cls.COLORS['end']}")
    
    @classmethod
    def params(cls, title: str, params: Dict[str, Any]):
        """Log parameters in a formatted way"""
        print(f"\n{cls.COLORS['yellow']}  {title}:{cls.COLORS['end']}")
        for key, value in params.items():
            if value is None:
                continue
            # Truncate long values
            str_val = str(value)
            if len(str_val) > 80:
                str_val = str_val[:77] + "..."
            print(f"    {key}: {str_val}")
    
    @classmethod
    def command(cls, cmd: List[str]):
        """Log a command being executed"""
        print(f"\n{cls.COLORS['bold']}  Command:{cls.COLORS['end']}")
        # Format command nicely - group by argument pairs
        formatted_parts = []
        i = 0
        while i < len(cmd):
            if i < len(cmd) - 1 and cmd[i].startswith('--'):
                # Argument with value
                formatted_parts.append(f"{cmd[i]} {cmd[i+1]}")
                i += 2
            else:
                formatted_parts.append(cmd[i])
                i += 1
        
        formatted = " \\\n      ".join(formatted_parts)
        print(f"    {formatted}")
        print()
    
    @classmethod
    def paths(cls, title: str, paths: Dict[str, Any]):
        """Log path information"""
        print(f"\n{cls.COLORS['blue']}  {title}:{cls.COLORS['end']}")
        for key, value in paths.items():
            if value is not None:
                print(f"    {key}: {value}")
    
    @classmethod
    def start_timer(cls, name: str):
        """Start a named timer"""
        cls._timers[name] = time.time()
        cls.info("TIMER", f"Started: {name}")
    
    @classmethod
    def stop_timer(cls, name: str, message: str = ""):
        """Stop a named timer and log elapsed time"""
        if name not in cls._timers:
            cls.warning("TIMER", f"Timer '{name}' was never started")
            return
        
        elapsed = time.time() - cls._timers[name]
        del cls._timers[name]
        
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s"
        
        msg = f"{message} " if message else ""
        print(f"{cls.COLORS['green']}[{cls.timestamp()}] ⏱ [{name}] {msg}({time_str}){cls.COLORS['end']}")
    
    @classmethod
    def timing(cls, category: str, start_time: float, message: str = ""):
        """Log timing from a start time"""
        elapsed = time.time() - start_time
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s"
        print(f"{cls.COLORS['green']}[{cls.timestamp()}] ⏱ [{category}] {message} ({time_str}){cls.COLORS['end']}")
    
    @classmethod
    def exception(cls, category: str, e: Exception):
        """Log full exception with traceback"""
        print(f"\n{cls.COLORS['red']}{'=' * 70}")
        print(f"  EXCEPTION in {category}")
        print(f"{'=' * 70}")
        print(f"  Error: {type(e).__name__}: {e}")
        print(f"\n  Traceback:")
        for line in traceback.format_exc().split('\n'):
            print(f"    {line}")
        print(f"{'=' * 70}{cls.COLORS['end']}\n")
    
    @classmethod
    def musubi_output(cls, line: str):
        """Log musubi output with special formatting"""
        line = line.rstrip()
        if not line:
            return
        
        # Highlight important musubi messages
        if 'error' in line.lower() or 'exception' in line.lower():
            print(f"{cls.COLORS['red']}  [MUSUBI] {line}{cls.COLORS['end']}")
        elif 'warning' in line.lower():
            print(f"{cls.COLORS['yellow']}  [MUSUBI] {line}{cls.COLORS['end']}")
        elif 'loading' in line.lower() or 'loaded' in line.lower():
            print(f"{cls.COLORS['cyan']}  [MUSUBI] {line}{cls.COLORS['end']}")
        elif '%|' in line:  # tqdm progress bar
            print(f"  [MUSUBI] {line}")
        else:
            print(f"  [MUSUBI] {line}")
    
    @classmethod
    def generation_summary(cls, gen_type: str, gen_id: str, params: Dict[str, Any], 
                          output_paths: Dict[str, Any], elapsed: float):
        """Log a complete generation summary"""
        cls.header(f"{gen_type.upper()} COMPLETE: {gen_id}")
        
        print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print()
        
        cls.params("Parameters Used", params)
        cls.paths("Output Files", output_paths)
        
        print(f"\n{cls.COLORS['green']}{'=' * 70}{cls.COLORS['end']}\n")


# Global instance for easy access
log = FukLogger()
