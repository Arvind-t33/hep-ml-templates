"""
Local installation utilities for hep-ml-templates.
Allows users to download blocks and configs to their project directory.
"""

import shutil
import os
from pathlib import Path
from typing import List, Dict, Set, Optional
import pkg_resources

# Mapping of extras to their corresponding blocks and configs
EXTRAS_TO_BLOCKS = {
    # Individual components
    'data-csv': {
        'blocks': ['ingest/csv_loader.py'],
        'core': ['interfaces.py', 'registry.py'],  # Add core dependencies
        'configs': ['data/csv_demo.yaml']
    },
    'data-higgs': {
        'blocks': ['ingest/csv_loader.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['data/higgs_uci.yaml'],
        'templates': ['preprocessors/preprocess_higgs_uci.py', 'preprocessors/config_higgs_uci.py']
    },
    'model-xgb': {
        'blocks': ['model/xgb_classifier.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/xgb_classifier.yaml']
    },
    'model-torch': {
        'blocks': ['model/ae_lightning.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/ae_lightning.yaml']
    },
    'model-gnn': {
        'blocks': ['model/gnn_pyg.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/gnn_pyg.yaml']
    },
    'preprocessing': {
        'blocks': ['preprocessing/standard_scaler.py', 'preprocessing/onehot_encoder.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['preprocessing/standard.yaml']
    },
    'evaluation': {
        'blocks': ['evaluation/classification_metrics.py', 'evaluation/reconstruction_metrics.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['evaluation/classification.yaml', 'evaluation/reconstruction.yaml']
    },
    
    # Algorithm-specific extras (combinations)
    'xgb': {
        'blocks': ['model/xgb_classifier.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/xgb_classifier.yaml', 'preprocessing/standard.yaml']
    },
    'torch': {
        'blocks': ['model/ae_lightning.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/ae_lightning.yaml', 'preprocessing/standard.yaml']
    },
    'gnn': {
        'blocks': ['model/gnn_pyg.py', 'preprocessing/standard_scaler.py'],
        'core': ['interfaces.py', 'registry.py'],
        'configs': ['model/gnn_pyg.yaml', 'preprocessing/standard.yaml']
    },
    
    # Complete pipeline bundles
    'pipeline-xgb': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'feature_eng/column_selector.py',
            'model/xgb_classifier.py',
            'evaluation/classification_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],  # Add all core modules for pipelines
        'configs': [
            'pipeline.yaml',
            'data/higgs_uci.yaml',
            'data/csv_demo.yaml',
            'preprocessing/standard.yaml',
            'feature_eng/column_selector.yaml',
            'model/xgb_classifier.yaml',
            'evaluation/classification.yaml',
            'runtime/local_cpu.yaml'
        ]
    },
    'pipeline-torch': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'model/ae_lightning.py',
            'evaluation/reconstruction_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],
        'configs': [
            'pipeline.yaml',
            'data/csv_demo.yaml',
            'preprocessing/standard.yaml',
            'model/ae_lightning.yaml',
            'evaluation/reconstruction.yaml',
            'runtime/local_gpu.yaml'
        ]
    },
    'pipeline-gnn': {
        'blocks': [
            'ingest/csv_loader.py',
            'preprocessing/standard_scaler.py',
            'model/gnn_pyg.py',
            'evaluation/classification_metrics.py'
        ],
        'core': ['interfaces.py', 'registry.py', 'config.py', 'utils.py'],
        'configs': [
            'pipeline.yaml',
            'data/custom_hep_example.yaml',
            'preprocessing/standard.yaml',
            'model/gnn_pyg.yaml',
            'evaluation/classification.yaml',
            'runtime/local_gpu.yaml'
        ]
    },
}

def get_package_path() -> Path:
    """Get the path to the installed hep-ml-templates package."""
    try:
        # Try to get the package path using the mlpipe module
        import mlpipe
        mlpipe_path = Path(mlpipe.__file__).parent  # This is src/mlpipe
        return mlpipe_path
    except:
        # Fallback: try to find it using pkg_resources
        try:
            import pkg_resources
            package_path = pkg_resources.resource_filename('mlpipe', '')
            return Path(package_path)
        except:
            raise FileNotFoundError("Could not locate hep-ml-templates installation")

def get_blocks_and_configs_for_extras(extras: List[str]) -> Dict[str, Set[str]]:
    """
    Given a list of extras, return the blocks, core modules, and configs that should be downloaded.
    
    Args:
        extras: List of extra names (e.g., ['model-xgb', 'data-higgs'])
    
    Returns:
        Dict with 'blocks', 'core', 'configs', and 'templates' keys containing sets of file paths
    """
    all_blocks = set()
    all_core = set()
    all_configs = set()
    all_templates = set()
    
    for extra in extras:
        if extra in EXTRAS_TO_BLOCKS:
            mapping = EXTRAS_TO_BLOCKS[extra]
            all_blocks.update(mapping.get('blocks', []))
            all_core.update(mapping.get('core', []))
            all_configs.update(mapping.get('configs', []))
            all_templates.update(mapping.get('templates', []))
        else:
            print(f"⚠️  Warning: Unknown extra '{extra}' - skipping")
    
    return {
        'blocks': all_blocks,
        'core': all_core,
        'configs': all_configs,
        'templates': all_templates
    }

def copy_core_modules(core_modules: Set[str], source_dir: Path, target_dir: Path):
    """Copy core modules from source to target directory."""
    core_source = source_dir / 'core'  # This is src/mlpipe/core
    
    if not core_source.exists():
        raise FileNotFoundError(f"Core directory not found: {core_source}")
    
    # Create the target core directory
    target_core = target_dir / 'core'
    target_core.mkdir(parents=True, exist_ok=True)
    
    for core_file in core_modules:
        source_file = core_source / core_file
        target_file = target_core / core_file
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied core module: {core_file}")
        else:
            print(f"⚠️  Warning: Core module not found: {source_file}")
    
    # Always copy __init__.py for the core module
    core_init = core_source / '__init__.py'
    target_init = target_core / '__init__.py'
    if core_init.exists():
        shutil.copy2(core_init, target_init)

def copy_blocks(blocks: Set[str], source_dir: Path, target_dir: Path):
    """Copy block files from source to target directory."""
    blocks_source = source_dir / 'blocks'
    
    if not blocks_source.exists():
        raise FileNotFoundError(f"Blocks directory not found: {blocks_source}")
    
    # Keep track of which module categories were installed
    installed_modules = {}
    
    for block_path in blocks:
        source_file = blocks_source / block_path
        target_file = target_dir / 'blocks' / block_path
        
        if source_file.exists():
            # Create target directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied block: {block_path}")
            
            # Track which categories were installed for __init__.py generation
            category = block_path.split('/')[0]  # e.g., 'ingest', 'model', 'preprocessing'
            module_name = Path(block_path).stem  # e.g., 'csv_loader', 'xgb_classifier'
            if category not in installed_modules:
                installed_modules[category] = []
            installed_modules[category].append(module_name)
            
            # Copy category-level __init__.py files
            category_init = blocks_source / category / '__init__.py'
            target_category_init = target_dir / 'blocks' / category / '__init__.py'
            if category_init.exists() and not target_category_init.exists():
                target_category_init.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(category_init, target_category_init)
        else:
            print(f"⚠️  Warning: Block file not found: {source_file}")
    
    # Create custom blocks/__init__.py that only imports installed blocks
    create_custom_blocks_init(installed_modules, target_dir / 'blocks' / '__init__.py')

def create_custom_blocks_init(installed_modules: Dict[str, List[str]], init_file_path: Path):
    """Create a custom __init__.py file for blocks that only imports installed modules."""
    
    init_content = ['# Auto-generated __init__.py for locally installed blocks']
    init_content.append('# Only imports the blocks that were actually installed')
    init_content.append('')
    
    for category, modules in installed_modules.items():
        for module in modules:
            init_content.append(f'from .{category} import {module}')
            # Add a comment about what this registers
            if category == 'ingest' and module == 'csv_loader':
                init_content[-1] += '                 # registers "ingest.csv"'
            elif category == 'model' and module == 'xgb_classifier':
                init_content[-1] += '              # registers "model.xgb_classifier"'
            elif category == 'preprocessing' and module == 'standard_scaler':
                init_content[-1] += '     # registers "preprocessing.standard_scaler"'
            elif category == 'feature_eng' and module == 'column_selector':
                init_content[-1] += '       # registers "feature.column_selector"'
            elif category == 'evaluation' and module == 'classification_metrics':
                init_content[-1] += ' # registers "eval.classification"'
    
    init_content.append('')
    
    # Write the custom __init__.py file
    init_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(init_file_path, 'w') as f:
        f.write('\n'.join(init_content))
    
    print(f"✅ Created custom blocks/__init__.py with {sum(len(modules) for modules in installed_modules.values())} imports")

def copy_configs(configs: Set[str], source_dir: Path, target_dir: Path):
    """Copy config files from source to target directory."""
    # The config files are in the hep-ml-templates root directory
    # source_dir is src/mlpipe, so we go up two levels to get to hep-ml-templates root
    config_source = source_dir.parent.parent / 'configs'  # src/mlpipe -> src -> hep-ml-templates -> configs
    
    if not config_source.exists():
        raise FileNotFoundError(f"Config directory not found: {config_source}")
    
    for config_path in configs:
        source_file = config_source / config_path
        target_file = target_dir / config_path
        
        if source_file.exists():
            # Create target directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied config: {config_path}")
        else:
            print(f"⚠️  Warning: Config file not found: {source_file}")

def copy_templates(templates: Set[str], source_dir: Path, target_dir: Path):
    """Copy template files from source to target directory."""
    # Templates are in src/mlpipe/templates/
    templates_source = source_dir / 'templates'
    
    if not templates_source.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_source}")
    
    for template_path in templates:
        source_file = templates_source / template_path
        target_file = target_dir / template_path
        
        if source_file.exists():
            # Create target directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied template: {template_path}")
        else:
            print(f"⚠️  Warning: Template file not found: {source_file}")

def install_local(extras: List[str], blocks_dir: str, configs_dir: Optional[str] = None) -> bool:
    """
    Install blocks and configs locally based on the provided extras.
    
    Args:
        extras: List of extra names to install
        blocks_dir: Directory where to install blocks
        configs_dir: Directory where to install configs (defaults to blocks_dir/configs)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🚀 Installing hep-ml-templates locally...")
        print(f"📦 Extras: {', '.join(extras)}")
        
        # Resolve target directories
        blocks_target = Path(blocks_dir).resolve()
        configs_target = Path(configs_dir).resolve() if configs_dir else blocks_target / 'configs'
        
        print(f"📁 Blocks will be installed to: {blocks_target}")
        print(f"⚙️  Configs will be installed to: {configs_target}")
        
        # Get source directory (installed package)
        package_path = get_package_path()
        print(f"📦 Source package: {package_path}")
        
        # Get blocks and configs to download
        to_download = get_blocks_and_configs_for_extras(extras)
        
        print(f"\n📋 Will download:")
        print(f"   🧩 {len(to_download['blocks'])} blocks")
        print(f"   🔧 {len(to_download['core'])} core modules")
        print(f"   ⚙️  {len(to_download['configs'])} configs")
        print(f"   📄 {len(to_download['templates'])} templates")
        
        # Copy core modules first (blocks depend on them)
        if to_download['core']:
            print(f"\n🔧 Copying core modules...")
            copy_core_modules(to_download['core'], package_path, blocks_target)
        
        # Copy blocks
        if to_download['blocks']:
            print(f"\n🧩 Copying blocks...")
            copy_blocks(to_download['blocks'], package_path, blocks_target)
        
        # Copy configs  
        if to_download['configs']:
            print(f"\n⚙️  Copying configs...")
            copy_configs(to_download['configs'], package_path, configs_target)
            
        # Copy templates
        if to_download['templates']:
            print(f"\n📄 Copying templates...")
            copy_templates(to_download['templates'], package_path, blocks_target)
        
        print(f"\n🎉 Local installation complete!")
        print(f"📁 Check your files:")
        print(f"   • Blocks: {blocks_target}")
        print(f"   • Configs: {configs_target}")
        if to_download['templates']:
            print(f"   • Templates: {blocks_target / 'preprocessors'}")
        print(f"\n💡 Next steps:")
        print(f"   1. Use the config files to load datasets")
        print(f"   2. Import and modify the preprocessor templates as needed")
        print(f"   3. Run your pipeline with minimal dataset-specific code!")
        
        return True
        
    except Exception as e:
        print(f"❌ Local installation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
