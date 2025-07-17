import sys
import os
import yaml
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

def validate_config(config):
    """Validate the configuration structure."""
    if not isinstance(config, dict):
        return False, "Configuration must be a dictionary"
    
    # Check rotate_x field if present
    if 'rotate_x' in config and not isinstance(config['rotate_x'], bool):
        return False, "'rotate_x' must be a boolean"
    
    # Check renames configuration if present
    if 'renames' in config:
        if not isinstance(config['renames'], dict):
            return False, "'renames' must be a dictionary"
        
        # Validate each rename mapping
        for old_path, new_path in config['renames'].items():
            if not isinstance(old_path, str) or not isinstance(new_path, str):
                return False, f"Rename mapping keys and values must be strings: '{old_path}' -> '{new_path}'"
            if not old_path.startswith('/') or not new_path.startswith('/'):
                return False, f"Rename paths must start with '/': '{old_path}' -> '{new_path}'"
    
    # Check prims configuration if present
    if 'prims' in config:
        if not isinstance(config['prims'], dict):
            return False, "'prims' must be a dictionary"
        
        # Validate each prim's configuration
        for prim_path, prim_config in config['prims'].items():
            if not isinstance(prim_config, dict):
                return False, f"Configuration for prim '{prim_path}' must be a dictionary"
            
            # Validate flags (collisions can be boolean or dict)
            for flag in ['visuals', 'rigid_body', 'articulation_root', 'translate_op']:
                if flag in prim_config and not isinstance(prim_config[flag], bool):
                    return False, f"'{flag}' flag for prim '{prim_path}' must be a boolean"
            
            # Special validation for collisions (can be boolean or dict)
            if 'collisions' in prim_config:
                collisions = prim_config['collisions']
                if not isinstance(collisions, (bool, dict)):
                    return False, f"'collisions' for prim '{prim_path}' must be a boolean or dictionary"
                if isinstance(collisions, dict):
                    if 'approximation' in collisions:
                        approx = collisions['approximation']
                        valid_approximations = ['convexHull', 'meshSimplification', 'triangleMesh', 'sphere', 'capsule', 'box']
                        if approx not in valid_approximations:
                            return False, f"'approximation' for collisions in prim '{prim_path}' must be one of {valid_approximations}"
            
            # Validate mass configuration if present
            if 'mass' in prim_config:
                mass = prim_config['mass']
                if not isinstance(mass, (int, float)) or mass <= 0:
                    return False, f"'mass' for prim '{prim_path}' must be a positive number"
            
            # Validate scale configuration if present
            if 'scale' in prim_config:
                scale = prim_config['scale']
                if not isinstance(scale, list) or len(scale) != 3:
                    return False, f"'scale' for prim '{prim_path}' must be a list of 3 numbers"
                for i, val in enumerate(scale):
                    if not isinstance(val, (int, float)):
                        return False, f"'scale' value {i} for prim '{prim_path}' must be a number"
            
            # Validate joint configuration if present
            if 'joint' in prim_config:
                joint_config = prim_config['joint']
                if not isinstance(joint_config, dict):
                    return False, f"Joint configuration for prim '{prim_path}' must be a dictionary"
    
    return True, None

def load_config(config_path):
    """Load and validate the YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return {
            'rotate_x': True,
            'prims': {}
        }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate configuration
        is_valid, error_msg = validate_config(config)
        if not is_valid:
            print(f"Error in configuration file: {error_msg}")
            print("Using default configuration.")
            return {
                'rotate_x': True,
                'prims': {}
            }
        
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        print("Using default configuration.")
        return {
            'rotate_x': True,
            'prims': {}
        }
    except Exception as e:
        print(f"Unexpected error loading config file: {e}")
        print("Using default configuration.")
        return {
            'rotate_x': True,
            'prims': {}
        }

def get_prim_config(prim_path, config):
    """Get the configuration for a specific prim."""
    if not config or 'prims' not in config:
        return {}
    
    prims_config = config.get('prims', {})
    return prims_config.get(prim_path, {})

def should_rotate_x(config):
    """Determine if x-axis rotation should be applied based on configuration."""
    if not config:
        return True
    
    return config.get('rotate_x', True)

def rotate_object_x_axis(stage, object_path, rotation_degrees):
    """
    Rotate an object around the x-axis by the specified degrees.
    
    Args:
        stage: USD stage
        object_path: Path to the object to rotate
        rotation_degrees: Rotation angle in degrees around x-axis
    """
    # Get the prim at the specified path
    prim = stage.GetPrimAtPath(object_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {object_path}")
        return False
    
    # Try to get existing Xform, or create one if it doesn't exist
    if prim.IsA(UsdGeom.Xform):
        xform = UsdGeom.Xform(prim)
    else:
        # If the prim is not an Xform, we need to check if it can be transformed
        # For geometry prims, we might need to work with their transform
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Error: Prim at {object_path} is not transformable")
            return False
        xform = xformable
    
    # Clear existing transform ops to avoid conflicts
    xform.ClearXformOpOrder()
    
    # Add rotation operation around x-axis
    # For clockwise rotation around x-axis: positive rotation in right-hand coordinate system
    rotate_op = xform.AddRotateXOp()
    rotate_op.Set(rotation_degrees)
    
    print(f"Applied {rotation_degrees} degree rotation around x-axis to {object_path}")
    return True

def apply_scale_transform(stage, object_path, scale_values):
    """
    Apply scale transformation to an object.
    
    Args:
        stage: USD stage
        object_path: Path to the object to scale
        scale_values: List of 3 scale factors [x, y, z]
    """
    # Get the prim at the specified path
    prim = stage.GetPrimAtPath(object_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {object_path}")
        return False
    
    # Try to get existing Xform, or create one if it doesn't exist
    if prim.IsA(UsdGeom.Xform):
        xform = UsdGeom.Xform(prim)
    else:
        # If the prim is not an Xform, we need to check if it can be transformed
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Error: Prim at {object_path} is not transformable")
            return False
        xform = xformable
    
    # Add scale operation
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale_values[0], scale_values[1], scale_values[2]))
    
    print(f"Applied scale {scale_values} to {object_path}")
    return True

def modify_usd_file_with_config(input_path, output_path, config, rotation_degrees=90.0):
    """
    Read a USD file, apply modifications based on config, and save to a new file.
    
    Args:
        input_path: Path to input USD file
        output_path: Path to output USD file
        config: Configuration dictionary from YAML file
        rotation_degrees: Rotation angle in degrees around x-axis (positive = clockwise)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        return False

    try:
        # Open the existing USD file
        stage = Usd.Stage.Open(input_path)
        if not stage:
            print(f"Error: Could not open USD file {input_path}")
            return False
        
        print(f"Successfully opened {input_path}")
        
        # modify each prims in config
        for prim_path, prim_config in config['prims'].items():
            if 'rigid_body' in prim_config and prim_config['rigid_body']:
                create_rigid_body(stage, prim_path)
            if 'articulation_root' in prim_config and prim_config['articulation_root']:
                create_articulation_root(stage, prim_path)
            if 'rotate_x' in prim_config and prim_config['rotate_x']:
                rotate_object_x_axis(stage, prim_path, rotation_degrees)
            if 'scale' in prim_config:
                apply_scale_transform(stage, prim_path, prim_config['scale'])
            if 'visuals' in prim_config and prim_config['visuals']:
                create_visuals(stage, prim_path)
            if 'collisions' in prim_config and prim_config['collisions']:
                create_collisions(stage, prim_path, prim_config['collisions'].get('approximation', 'convexHull'))
            if 'translate_op' in prim_config and prim_config['translate_op']:
                create_translate_op(stage, prim_path)
            if 'mass' in prim_config:
                create_mass(stage, prim_path, prim_config['mass'])
            if 'joint' in prim_config and prim_config['joint']:
                create_joint(stage, prim_path, prim_config['joint'])
            if 'transform_to_translate_orient_scale' in prim_config and prim_config['transform_to_translate_orient_scale']:
                transform_to_translate_orient_scale(stage, prim_path)
            if 'add_transform' in prim_config and prim_config['add_transform']:
                add_transform(stage, prim_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export the modified stage to the new file
        stage.GetRootLayer().Export(output_path)
        
        print(f"Successfully saved modified USD to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during USD modification: {e}")
        return False

def create_rigid_body(stage, prim_path):
    """Apply rigid body physics API to a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    # For instances (which can come from references), we apply the API directly to the instance
    # rather than trying to get the prototype
    if UsdPhysics.RigidBodyAPI.CanApply(prim):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        print(f"Applied RigidBodyAPI to {prim_path}")
        return True
    else:
        print(f"Warning: Cannot apply RigidBodyAPI to {prim_path}")
        return False
    
@DeprecationWarning
def create_articulation_root(stage, prim_path):
    """Apply articulation root API to a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    if UsdPhysics.ArticulationRootAPI.CanApply(prim):
        articulation_root_api = UsdPhysics.ArticulationRootAPI.Apply(prim)
        print(f"Applied ArticulationRootAPI to {prim_path}")
        prim.GetAttribute("physxArticulation:solverPositionIterationCount").Set(16)
        return True
    else:
        print(f"Warning: Cannot apply ArticulationRootAPI to {prim_path}")
        return False

def create_visuals(stage, prim_path):
    """Enable visual representation for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    # For visuals, we typically ensure the prim is visible and has proper purpose
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)
        # Set visibility to inherited (default visible state)
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        # Set purpose to default for rendering
        imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
        print(f"Enabled visuals for {prim_path}")
        return True
    else:
        print(f"Warning: Prim at {prim_path} is not imageable")
        return False

def create_collisions(stage, prim_path, approximation="convexHull"):
    """Apply collision physics APIs to a prim.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim
        approximation: Collision approximation method ("convexHull", "meshSimplification", "triangleMesh", etc.)
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    # Apply collision API
    if UsdPhysics.CollisionAPI.CanApply(prim):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        print(f"Applied CollisionAPI to {prim_path}")
        
        # If it's a mesh, also apply mesh collision API
        if prim.IsA(UsdGeom.Mesh):
            if UsdPhysics.MeshCollisionAPI.CanApply(prim):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision_api.CreateApproximationAttr().Set(approximation)
                print(f"Applied MeshCollisionAPI to {prim_path} with {approximation} approximation")
        
        return True
    else:
        print(f"Warning: Cannot apply CollisionAPI to {prim_path}")
        return False

def get_prim_paths(body0_path, body1_path):
    """Get the prim paths from body0 to body1."""
    prim_path_list = []
    
    # 以'/'分割路径，过滤空字符串
    body0_parts = [part for part in body0_path.split('/') if part]
    body1_parts = [part for part in body1_path.split('/') if part]
    
    # 找到共同前缀的长度
    common_prefix_len = 0
    for i in range(min(len(body0_parts), len(body1_parts))):
        if body0_parts[i] == body1_parts[i]:
            common_prefix_len = i + 1
        else:
            break
    
    # 添加body0路径
    prim_path_list.append(body0_path)
    
    # 从body0向上到共同祖先的路径
    for i in range(len(body0_parts), common_prefix_len, -1):
        path_parts = body0_parts[:i - 1]
        if path_parts:
            prim_path_list.append('/' + '/'.join(path_parts))
    
    # 删除公共父路径
    prim_path_list.pop(-1)
    
    # 从共同祖先向下到body1的路径
    for i in range(common_prefix_len + 1, len(body1_parts) + 1):
        path_parts = body1_parts[:i]
        if path_parts:
            prim_path_list.append('/' + '/'.join(path_parts))
    
    return prim_path_list

def create_joint(stage, prim_path, joint_config):
    """Create a joint based on the joint configuration."""
    if not joint_config or 'type' not in joint_config:
        print(f"Error: Invalid joint configuration for {prim_path}")
        return False
    
    joint_type = joint_config['type']
    joint_name = joint_config.get('name', 'Joint')
    
    # Create joint path under the prim
    joint_path = Sdf.Path(f"{prim_path}/{joint_name}")
    
    if joint_type == 'prismatic':
        if 'body1' not in joint_config:
            print(f"Error: Prismatic joint for {prim_path} missing body0 or body1")
            return False
        
        # Create prismatic joint
        prismatic_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        
        # Set body relationships
        body0_path = Sdf.Path(prim_path if 'body0' not in joint_config else joint_config['body0'])
        body1_path = Sdf.Path(joint_config['body1'])
        
        prismatic_joint.CreateBody0Rel().SetTargets([body0_path])
        prismatic_joint.CreateBody1Rel().SetTargets([body1_path])
        
        # Set joint axis if provided
        if 'axis' in joint_config:
            axis = joint_config['axis']
            # Convert to GfVec3f if it's a list/array
            if isinstance(axis, (list, tuple)):
                axis = Gf.Vec3f(axis[0], axis[1], axis[2])
            prismatic_joint.CreateAxisAttr().Set(axis)
        
        # Set local position and rotation if provided (overrides calculated value)
        if 'local_position_0' in joint_config and 'local_rotation_0' in joint_config:
            local_pos = joint_config['local_position_0']
            # Convert to GfVec3f if it's a list/array
            if isinstance(local_pos, (list, tuple)):
                local_pos = Gf.Vec3f(local_pos[0], local_pos[1], local_pos[2])
            prismatic_joint.CreateLocalPos0Attr().Set(local_pos)
            print(f"Overriding calculated position with config value: {local_pos}")
            
            local_rot = joint_config['local_rotation_0']
            # Convert to GfQuatf if it's a list/array (assuming Euler angles in degrees)
            if isinstance(local_rot, (list, tuple)):
                if len(local_rot) == 3:
                    # Convert Euler angles (x, y, z) in degrees to quaternion
                    euler_x, euler_y, euler_z = local_rot
                    # Create rotation matrix from Euler angles and convert to quaternion
                    rot_matrix = (Gf.Matrix3d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), euler_x)) *
                                  Gf.Matrix3d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), euler_y)) *
                                  Gf.Matrix3d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), euler_z)))
                    local_rot = Gf.Quatf(rot_matrix.ExtractRotation().GetQuat())
                elif len(local_rot) == 4:
                    # Already a quaternion (w, x, y, z)
                    local_rot = Gf.Quatf(local_rot[0], local_rot[1], local_rot[2], local_rot[3])
            prismatic_joint.CreateLocalRot0Attr().Set(local_rot)
            print(f"Overriding calculated rotation with config value: {local_rot}")
        else:
            # Calculate local position and rotation from body transforms
            prim_path_list = get_prim_paths(body0_path.pathString, body1_path.pathString)
            body_prim_list = [stage.GetPrimAtPath(path) for path in prim_path_list]
            
            if all(prim.IsValid() for prim in body_prim_list):
                # Get transform matrices
                body_xformable_list = [UsdGeom.Xformable(prim) for prim in body_prim_list]
                
                if all(xformable for xformable in body_xformable_list):
                    body_matrix_list = [xformable.GetLocalTransformation() for xformable in body_xformable_list]
                    
                    # Calculate relative transforms in reverse order
                    # Start from the last transformation and work backwards
                    relative_transform = body_matrix_list[-1]
                    
                    # Process matrix list in reverse order
                    for i in range(len(body_matrix_list) - 2, -1, -1):
                        body_prev_matrix = body_matrix_list[i]
                        
                        # Calculate relative transform: current_transform * inverse(previous_transform)
                        body_prev_inverse = body_prev_matrix.GetInverse()
                        relative_transform = relative_transform * body_prev_inverse
                        
                        print(f"Step {len(body_matrix_list) - i - 1}: Relative transform calculated")
                    
                    # Extract translation and rotation from final relative transform
                    relative_translation = relative_transform.ExtractTranslation()
                    relative_rotation_quat = relative_transform.ExtractRotationQuat()
                    
                    # Set local position and rotation
                    prismatic_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(relative_translation))
                    prismatic_joint.CreateLocalRot0Attr().Set(Gf.Quatf(relative_rotation_quat.GetReal(), 
                                                                       *relative_rotation_quat.GetImaginary()))
                    
                    print(f"Final joint transform (reverse order) - Position: {relative_translation}, Rotation: {relative_rotation_quat}")
                else:
                    print("Warning: Could not get xformable for body0 or body1")
            else:
                print(f"Warning: Could not find body0 ({body0_path}) or body1 ({body1_path}) prims")
        
        # Set joint limits if provided
        if 'lower_limit' in joint_config:
            prismatic_joint.CreateLowerLimitAttr().Set(joint_config['lower_limit'])
        if 'upper_limit' in joint_config:
            prismatic_joint.CreateUpperLimitAttr().Set(joint_config['upper_limit'])
        
        print(f"Created prismatic joint {joint_name} at {prim_path}")
        return True
    else:
        print(f"Error: Unsupported joint type '{joint_type}' for {prim_path}")
        return False

def create_translate_op(stage, prim_path):
    """Apply translation transform operation to a prim.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim to translate
        
    Returns:
        bool: True if successful, False otherwise
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    try:
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Error: Prim at {prim_path} is not transformable")
            return False
            
        translate_op = xformable.GetTranslateOp()
        if not translate_op:
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(0, 0, 0))  # Set to origin or appropriate position
        print(f"Applied translation operation to {prim_path}")
        return True
        
    except Exception as e:
        print(f"Error creating translate op for {prim_path}: {e}")
        return False

def create_mass(stage, prim_path, mass_value):
    """Apply mass properties to a prim.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim to set mass for
        mass_value: Mass value in kg
        
    Returns:
        bool: True if successful, False otherwise
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    try:
        # Apply mass API to the prim
        if UsdPhysics.MassAPI.CanApply(prim):
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(mass_value)
            print(f"Applied mass {mass_value} kg to {prim_path}")
            return True
        else:
            print(f"Warning: Cannot apply MassAPI to {prim_path}")
            return False
            
    except Exception as e:
        print(f"Error creating mass for {prim_path}: {e}")
        return False

def get_references_from_prim(stage, prim_path):
    """
    Get all references from a prim at the given path.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim
        
    Returns:
        list: List of reference asset paths, or empty list if no references
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return []
    
    # Get the prim's references
    references = prim.GetReferences()
    
    reference_paths = []
    try:
        # Get the prim spec from the current edit target
        prim_spec = stage.GetEditTarget().GetPrimSpecForScenePath(prim.GetPath())
        if prim_spec and prim_spec.referenceList:
            for ref in prim_spec.referenceList.addedItems:
                if ref.assetPath:
                    reference_paths.append(ref.assetPath)
    except Exception as e:
        print(f"Warning: Could not get references for {prim_path}: {e}")
    
    return reference_paths

def check_prim_has_references(stage, prim_path):
    """
    Check if a prim has any references.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim
        
    Returns:
        bool: True if prim has references, False otherwise
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    
    try:
        # Get the prim spec from the current edit target
        prim_spec = stage.GetEditTarget().GetPrimSpecForScenePath(prim.GetPath())
        if prim_spec and prim_spec.referenceList:
            return len(prim_spec.referenceList.addedItems) > 0
    except Exception:
        pass
    
    return False

def add_reference_to_prim(stage, prim_path, reference_asset_path, prim_path_in_reference=None):
    """
    Add a reference to a prim.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim to add reference to
        reference_asset_path: Path to the asset to reference
        prim_path_in_reference: Optional path within the referenced file
        
    Returns:
        bool: True if successful, False otherwise
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    try:
        references = prim.GetReferences()
        if prim_path_in_reference:
            references.AddReference(reference_asset_path, prim_path_in_reference)
        else:
            references.AddReference(reference_asset_path)
        
        print(f"Added reference '{reference_asset_path}' to prim '{prim_path}'")
        return True
        
    except Exception as e:
        print(f"Error adding reference to {prim_path}: {e}")
        return False

def get_all_prims_with_references(stage):
    """
    Get all prims in the stage that have references.
    
    Args:
        stage: USD stage
        
    Returns:
        list: List of prim paths that have references
    """
    prims_with_refs = []
    
    for prim in stage.Traverse():
        if check_prim_has_references(stage, prim.GetPath()):
            prims_with_refs.append(str(prim.GetPath()))
    
    return prims_with_refs

def transform_to_translate_orient_scale(stage, prim_path):
    """
    Convert a prim's transform matrix to explicit Translate, Orient, and Scale operations.
    
    This function takes a prim's existing transform matrix and decomposes it into
    separate translate, orient (rotation), and scale operations, then applies them
    to the prim in the standard TRS order.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim to transform
        
    Returns:
        bool: True if successful, False otherwise
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    try:
        # Get the xformable prim
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Error: Prim at {prim_path} is not transformable")
            return False
        
        # Get the current local transform matrix
        matrix = xformable.GetLocalTransformation()
        
        # Decompose the matrix into translation, rotation, and scale
        translate = matrix.ExtractTranslation()
        rotationd = matrix.ExtractRotationQuat()  # 四元数
        scale = Gf.Vec3d(
            Gf.Vec3d(matrix[0][0], matrix[0][1], matrix[0][2]).GetLength(),
            Gf.Vec3d(matrix[1][0], matrix[1][1], matrix[1][2]).GetLength(),
            Gf.Vec3d(matrix[2][0], matrix[2][1], matrix[2][2]).GetLength()
        )
        
        # 1. Clear existing transform operations first
        xformable.ClearXformOpOrder()
        
        # 2. Add translate operation
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(translate)
        
        # 3. Add rotation operation
        rotationf = Gf.Quatf(rotationd.GetReal(), *rotationd.GetImaginary())
        orient_op = xformable.AddOrientOp()
        orient_op.Set(rotationf)
        
        # 4. Add scale operation
        scale_op = xformable.AddScaleOp()
        scale_op.Set(scale)
        
        print(f"Successfully decomposed transform matrix to TRS operations for {prim_path} with scale {scale}")
        
        return True
        
    except Exception as e:
        print(f"Error decomposing transform matrix for {prim_path}: {e}")
        return False

def add_transform(stage, prim_path):
    """
    Add standard translate, orient, and scale transform operations to a prim.
    
    This function sets up a prim with the standard TRS (Translate, Rotate, Scale)
    transform operations using default values (identity transform).
    
    Args:
        stage: USD stage
        prim_path: Path to the prim to add transforms to
        
    Returns:
        bool: True if successful, False otherwise
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: No prim found at path {prim_path}")
        return False
    
    try:
        # Get the xformable prim
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Error: Prim at {prim_path} is not transformable")
            return False
        
        # Clear existing transform operations first
        xformable.ClearXformOpOrder()
        
        # Add translate operation with default (0, 0, 0)
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        
        # Add orient operation with identity quaternion
        orient_op = xformable.AddOrientOp()
        orient_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # Identity quaternion
        
        # Add scale operation with default (1, 1, 1)
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
        
        print(f"Successfully added TRS transform operations to {prim_path}")
        
        return True
        
    except Exception as e:
        print(f"Error adding transform operations to {prim_path}: {e}")
        return False

def main():
    """Main function to handle command line arguments and execute the modification."""
    if len(sys.argv) < 4:
        print("Usage: python usd_modification.py base_dir input.usd output.usd [config.yaml] [rotation_degrees]")
        print("Example: python usd_modification.py assets/robots/v2 resaved.usd modified.usd configs/structure_config.yaml")
        print("Example: python usd_modification.py assets/robots/v2 resaved.usd modified.usd configs/structure_config.yaml 90")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    input_file = os.path.join(base_dir, sys.argv[2])
    output_file = os.path.join(base_dir, sys.argv[3])
    config_path = os.path.join(base_dir, sys.argv[4])
    config = load_config(config_path)
    
    # Optional rotation degrees argument
    rotation_degrees = float(sys.argv[5]) if len(sys.argv) > 5 else 90.0
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    if config_path:
        print(f"Config file: {config_path}")
        print(f"Rotate X enabled: {should_rotate_x(config)}")
        prims_config = config.get('prims', {})
        if prims_config:
            print(f"Configured prims: {list(prims_config.keys())}")
    print(f"Rotation: {rotation_degrees} degrees around x-axis")
    
    success = modify_usd_file_with_config(input_file, output_file, config, rotation_degrees)
    
    if success:
        print("USD modification completed successfully!")
    else:
        print("USD modification failed!")


if __name__ == "__main__":
    main() 