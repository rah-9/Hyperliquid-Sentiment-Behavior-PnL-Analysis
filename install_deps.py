"""
Quick script to install all required dependencies.
Run this if you're getting import errors.
"""
import subprocess
import sys

def install_package(package):
	"""Install a package using pip."""
	try:
		subprocess.check_call([sys.executable, "-m", "pip", "install", package])
		print(f"✓ Installed {package}")
		return True
	except subprocess.CalledProcessError:
		print(f"✗ Failed to install {package}")
		return False

def main():
	print("=" * 60)
	print("Installing Dependencies for Hyperliquid Sentiment Analysis")
	print("=" * 60)
	print()
	
	# Read requirements
	try:
		with open('requirements.txt', 'r') as f:
			packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
	except FileNotFoundError:
		print("Error: requirements.txt not found!")
		return
	
	# Install each package
	success_count = 0
	for package in packages:
		if install_package(package):
			success_count += 1
	
	print()
	print("=" * 60)
	print(f"Installation complete: {success_count}/{len(packages)} packages installed")
	print("=" * 60)
	print()
	print("You can now:")
	print("  1. Run the analysis: python scripts/run_all.py")
	print("  2. Launch the web app: streamlit run app.py")

if __name__ == "__main__":
	main()

