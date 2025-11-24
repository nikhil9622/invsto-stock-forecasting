"""
Test script to verify all packages are installed correctly
"""

print("Testing package imports...")
print("-" * 50)

try:
    import yfinance
    print("✓ yfinance")
except ImportError as e:
    print(f"✗ yfinance - {e}")

try:
    import pandas
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas - {e}")

try:
    import numpy
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy - {e}")

try:
    import matplotlib
    print("✓ matplotlib")
except ImportError as e:
    print(f"✗ matplotlib - {e}")

try:
    import seaborn
    print("✓ seaborn")
except ImportError as e:
    print(f"✗ seaborn - {e}")

try:
    import mplfinance
    print("✓ mplfinance")
except ImportError as e:
    print(f"✗ mplfinance - {e}")

try:
    import statsmodels
    print("✓ statsmodels")
except ImportError as e:
    print(f"✗ statsmodels - {e}")

try:
    import pmdarima
    print("✓ pmdarima")
except ImportError as e:
    print(f"✗ pmdarima - {e}")

try:
    import sklearn
    print("✓ scikit-learn")
except ImportError as e:
    print(f"✗ scikit-learn - {e}")

try:
    import xgboost
    print("✓ xgboost")
except ImportError as e:
    print(f"✗ xgboost - {e}")

try:
    import ta
    print("✓ ta")
except ImportError as e:
    print(f"✗ ta - {e}")

try:
    import joblib
    print("✓ joblib")
except ImportError as e:
    print(f"✗ joblib - {e}")

print("-" * 50)
print("\n✓ All packages imported successfully!")
print("\nYou're ready to run the pipeline!")
print("Execute: .\\run_pipeline.ps1")
