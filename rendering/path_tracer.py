is_to_output = False
useGUI = True
image_output_folder = 'results2/'
obj_input_folder = '../simulation/results/'

from luisa import *
from luisa.builtin import *
from luisa.types import *
from luisa.util import *
from pywavefront import Wavefront
 
import cv2, os # for video
import cornell_box
import numpy as np

init()

Material = StructType(
    albedo=float3, # 物体颜色，在shininess!=0情况下表示specular系数
    emission=float3,
    shininess=float3
    # shininess.x 镜面系数，支持glossy，0表示diffuse
    # shininess.y index of refraction，折射率，-1表示不透明
    # shininess.z 材质，1 : Diffuse, 2 : glossy, 3: perfect mirror, 4: perfect fluid
    )
Onb = StructType(tangent=float3, binormal=float3, normal=float3)

heap = BindlessArray()
accel = Accel()
vertex_buffer = Buffer(1, float3) # 未初始化
material_buffer = Buffer(1, float3) # 未初始化

mesh_cnt = 0
obj_mesh_cnt = 0

res = 1024, 1024
image = Image2D(*res, 4, float)
accum_image = Image2D(*res, 4, float)
seed_image = Image2D(*res, 1, uint, storage="INT")
ldr_image = Image2D(*res, 4, float, storage="BYTE")

frame_index = 0

@func
def to_world(self, v: float3):
    return v.x * self.tangent + v.y * self.binormal + v.z * self.normal


Onb.add_method(to_world, "to_world")


@func
def linear_to_srgb(x: float3):
    return clamp(select(1.055 * x ** (1.0 / 2.4) - 0.055,
                        12.92 * x,
                        x <= 0.00031308),
                 0.0, 1.0)


@func
def make_onb(normal: float3):
    binormal = normalize(select(
        float3(0.0, -normal.z, normal.y),
        float3(-normal.y, normal.x, 0.0),
        abs(normal.x) > abs(normal.z)))
    tangent = normalize(cross(binormal, normal))
    result = Onb()
    result.tangent = tangent
    result.binormal = binormal
    result.normal = normal
    return result


@func
def generate_ray(p):
    fov = 27.8 / 180 * 3.1415926
    origin = float3(-0.01, 0.995, 5.0)
    pixel = origin + float3(p * tan(0.5 * fov), -1.0)
    direction = normalize(pixel - origin)
    return make_ray(origin, direction, 0.0, 1e30)


@func
def cosine_sample_hemisphere(u: float2, shininess: float):
    cosTheta = pow(1.0 - u.x, 1.0 / (1.0 + shininess))
    sinTheta = sqrt(1.0 - cosTheta * cosTheta)
    phi = 2.0 * 3.1415926 * u.y
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)


@func
def balanced_heuristic(pdf_a, pdf_b):
    return pdf_a / max(pdf_a + pdf_b, 1e-4)

@func
def specular_brdf(shininess: float, specular: float3, w_i: float3, w_o: float3, n: float3):
    # Normalize vectors
    w_i = normalize(-w_i)
    w_o = normalize(w_o)
    n = normalize(n)

    # Calculate halfway vector
    h = normalize(w_i + w_o)

    # Calculate the dot product between normal and halfway vector
    n_dot_h = max(dot(n, h), 0)

    return specular * (n_dot_h ** shininess)


@func
def reflectance(cosTheta: float, refraction_ratio: float):
    # Schlick近似Fresnel
    FresnelR0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio)
    FresnelR0 *= FresnelR0
    return FresnelR0 + (1.0 - FresnelR0) * pow(1.0 - cosTheta, 5.0)


@func
def raytracing_kernel(image, seed_image, accel, heap, resolution, vertex_buffer, material_buffer, mesh_cnt, frame_index):
    set_block_size(8, 8, 1)
    coord = dispatch_id().xy
    frame_size = float(min(resolution.x, resolution.y))
    if frame_index == 0:
        sampler = make_random_sampler(coord.x, coord.y)
    else:
        sampler = RandomSampler(seed_image.read(coord))
    radiance = float3(0)
    light_position = float3(-0.24, 1.98, 0.16)
    light_u = float3(-0.24, 1.98, -0.22) - light_position
    light_v = float3(0.23, 1.98, 0.16) - light_position
    light_emission = float3(17.0, 12.0, 4.0)
    light_area = length(cross(light_u, light_v))
    light_normal = normalize(cross(light_u, light_v))
    rx = sampler.next()
    ry = sampler.next()
    pixel = (float2(coord) + float2(rx, ry)) / frame_size * 2.0 - 1.0
    ray = generate_ray(pixel * float2(1.0, -1.0))
    beta = float3(1.0)
    pdf_bsdf = 1e30
    mis_weight = float(1.0)
    mis_used = False
    for depth in range(100): # 用russian routelette终止
        # closest shader
        hit = accel.trace_closest(ray, -1)
        # 1. miss shader: 由于没有环境贴图，直接break
        if hit.miss():
            break
        # 2. 读取shading point信息
        i0 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 0)
        i1 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 1)
        i2 = heap.buffer_read(int, hit.inst, hit.prim * 3 + 2)
        p0 = vertex_buffer.read(i0)
        p1 = vertex_buffer.read(i1)
        p2 = vertex_buffer.read(i2)
        p = hit.interpolate(p0, p1, p2)
        n = normalize(cross(p1 - p0, p2 - p0))
        pp = offset_ray_origin(p, n) # p向n偏移一些防止自交
        w_i = normalize(ray.get_dir())
        cos_wi = dot(-w_i, n) # 入射方向与法线夹角cos
        material = Material()
        if hit.inst >= mesh_cnt:
            material.albedo = material_buffer.read(mesh_cnt * 3 + 0)
            material.emission = material_buffer.read(mesh_cnt * 3 + 1)
            material.shininess = material_buffer.read(mesh_cnt * 3 + 2)
        else:
            material.albedo = material_buffer.read(hit.inst * 3 + 0)
            material.emission = material_buffer.read(hit.inst * 3 + 1)
            material.shininess = material_buffer.read(hit.inst * 3 + 2)
        shininess = 1.0 # material.shininess.x
        index_of_refraction = material.shininess.y
        material_type = material.shininess.z
        
        is_backface_hit = cos_wi <= 0
        if is_backface_hit:
            if material_type != 4.0: # not fluid
                break
            cos_wi = -cos_wi
            n = -n
            
        if material_type == 3.0: # perfect mirror
            # 直接使用下一个交点
            new_direction = reflect(w_i, n)
            ray = make_ray(pp, new_direction, 0.0, 1e30)
            beta *= material.albedo # cos_wi
            mis_used = False
            continue
        elif material_type == 4.0: # perfect fluid
            # refraction_ratio = index_of_refraction
            # if is_backface_hit:
            #     refraction_ratio = 1.0 / index_of_refraction
            refraction_ratio = select(index_of_refraction, 1.0 / index_of_refraction,  is_backface_hit)
            surface_reflectance = reflectance(cos_wi, refraction_ratio)
            sin_wi = sqrt(1.0 - cos_wi * cos_wi)
            is_total_reflection = refraction_ratio * sin_wi > 1.0
            pp = offset_ray_origin(p, n) # p向n偏移一些防止自交
            new_direction = reflect(w_i, n)
            if sampler.next() > surface_reflectance and not is_total_reflection:
                # refraction
                pp = offset_ray_origin(p, -n) # p向n偏移一些防止自交
                direction_parallel = refraction_ratio * (w_i + cos_wi * n)
                direction_perp = -sqrt(abs(1.0 - length_squared(direction_parallel))) * n
                new_direction = normalize(direction_parallel + direction_perp)
            ray = make_ray(pp, new_direction, 0.0, 1e30)
            mis_used = False
            continue

        # Next Event Estimation

        # 2-1. BRDF direct：如果BRDF采样击中了光源
        if any(material.emission != float3(0,0,0)):
            mis_weight = 1.0
            if mis_used:
                d_light = length(p - ray.get_origin())
                pdf_light = d_light * d_light / (light_area * cos_wi)
                mis_weight = balanced_heuristic(pdf_bsdf, pdf_light) # bsdf mis
            radiance += mis_weight * beta * light_emission # beta包括上一步完成的光传播系数
            break

        # 2-2. BRDF indirect，采样策略一：对diffuse场景点进行cos-weighted采样，其radiance贡献放到下一层循环进行估计
        onb = make_onb(n)
        ux = sampler.next()
        uy = sampler.next()
        local_direction = cosine_sample_hemisphere(float2(ux, uy), shininess)
        new_direction = onb.to_world(local_direction)
        if material_type == 2.0: # glossy brdf
            lobe_direction = reflect(w_i, n)
            onb_lobe = make_onb(lobe_direction)
            new_direction = normalize(onb_lobe.to_world(local_direction))

        # 2-3. 采样策略二：对光源采样，构造一条从shading point到光源采样点的shadow ray
        # 2-3-1. 构建shadow ray并且进行追踪，查看是否有遮挡
        mis_used = True
        ux_light = sampler.next()
        uy_light = sampler.next()
        p_light = light_position + ux_light * light_u + uy_light * light_v
        pp_light = offset_ray_origin(p_light, light_normal)
        d_light = length(pp - pp_light)
        wi_light = normalize(pp_light - pp)
        shadow_ray = make_ray(offset_ray_origin(pp, n), wi_light, 0.0, d_light)
        occluded = accel.trace_any(shadow_ray, -1)
        cos_wi_light = dot(wi_light, n)
        cos_light = -dot(light_normal, wi_light)
        pdf_light = (d_light * d_light) / (light_area * cos_light)
        pdf_bsdf = cos_wi_light * (1 / 3.1415926)
        if material_type == 2.0:
            lobe_direction = normalize(reflect(w_i, n))
            pdf_bsdf = pow(dot(lobe_direction, wi_light), shininess) * (1 + shininess) / 2 / 3.1415926
        mis_weight = balanced_heuristic(pdf_light, pdf_bsdf) # light mis
        # 2-3-2. 如果无遮挡，且非backface hit，则计入直接光照radiance
        if ((not occluded and cos_wi_light > 0) and cos_light > 0):
            # 加入radiance贡献
            bsdf = float3(material.albedo * (1 / 3.1415926) * cos_wi_light)
            if material_type == 2.0:
                bsdf = specular_brdf(shininess, material.albedo / 3.1415926, w_i, new_direction, n) * cos_wi_light
            radiance += beta * bsdf * mis_weight * \
                light_emission / pdf_light
                
        pdf_bsdf = local_direction.z ** shininess * (1 + shininess) / 2 / 3.1415926
        # 2-4. 下一步追踪的光线，其权重应当是brdf * mis_brdf
        ray = make_ray(pp, new_direction, 0.0, 1e30)
        if material_type == 1.0: # diffuse
            beta *= material.albedo
        elif material_type == 2.0: # glossy
            cos_wo = dot(new_direction, n)
            beta *= specular_brdf(shininess, material.albedo / 3.1415926, w_i, new_direction, n) * (cos_wo / pdf_bsdf)
        
        # 2-5. russian routelette
        l = dot(float3(0.212671, 0.715160, 0.072169), beta)
        if l == 0.0:
            break
        q = max(l, 0.05)
        r = sampler.next()
        if r >= q:
            break
        beta *= float(1.0 / q)
        if any(isnan(radiance)):
            radiance = float3(0.0)
    seed_image.write(coord, sampler.state)
    image.write(coord, float4(
        clamp(radiance, 0.0, 30.0), 1.0))


@func
def accumulate_kernel(accum_image, curr_image):
    p = dispatch_id().xy
    accum = accum_image.read(p)
    curr = curr_image.read(p).xyz
    t = 1.0 / (accum.w + 1.0)
    accum_image.write(p, float4(lerp(accum.xyz, curr, t), accum.w + 1.0))


@func
def aces_tonemapping(x: float3):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


@func
def clear_kernel(image):
    image.write(dispatch_id().xy, float4(0.0))


@func
def hdr2ldr_kernel(hdr_image, ldr_image, scale: float):
    coord = dispatch_id().xy
    hdr = hdr_image.read(coord)
    ldr = linear_to_srgb(hdr.xyz * scale)
    ldr = aces_tonemapping(ldr)
    ldr_image.write(coord, float4(ldr, 1.0))


def init_buffer(obj):
    global vertex_buffer, material_buffer
    obj_v = [[(v[0] - 1) / 15 - 1, v[1] / 16 - 0.2 , (v[2] - 1) / 15 - 1] for v in obj.vertices]
    vertex_buffer = Buffer(len(cornell_box.vertices) + len(obj_v), float3)
    vertex_arr = [[*item, 0.0] for item in cornell_box.vertices]
    vertex_arr += [[*item, 0.0] for item in obj_v]
    vertex_arr = np.array(vertex_arr, dtype=np.float32)
    vertex_buffer.copy_from(vertex_arr)
    material_arr = [
        [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0],
        [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0],
        [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0],
        [0.14, 0.45, 0.091, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 2.0, 0.0],
        [0.63, 0.065, 0.05, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 3.0, 0.0],
        # [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0],
        # [0.725, 0.71, 0.68, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0],
        [0.0,   0.0, 0.0, 0.0], [17.0, 12.0, 4.0, 0.0], [1.0, -1.0, 1.0, 0.0],
        [0.7, 0.7, 0.7,  0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.3333, 4.0, 0.0]
    ]
    material_buffer = Buffer(len(material_arr), float3)
    material_arr = np.array(material_arr, dtype=np.float32)
    material_buffer.copy_from(material_arr)


# def update_heap(obj):
#     global mesh_cnt, accel, heap
#     accel.pop()
#     heap.remove_buffer(mesh_cnt)
#     indices = [v + len(cornell_box.vertices) for face in obj.mesh_list[0].faces for v in face]
#     triangle_buffer = Buffer(len(indices),int)
#     triangle_buffer.copy_from(np.array(indices, dtype=np.int32))
#     heap.emplace(mesh_cnt, triangle_buffer)
#     accel.add(vertex_buffer, triangle_buffer, visibility_mask = 64)
#     accel.update()
#     heap.update()


def init_heap(obj):
    global mesh_cnt, accel, heap
    mesh_cnt = 0
    accel = Accel.empty()
    heap = BindlessArray.empty()

    for mesh in cornell_box.faces:
        indices = []
        for item in mesh:
            assert (len(item) == 4)
            for x in [0, 1, 2, 0, 2, 3]:
                indices.append(item[x])
        if mesh_cnt == 5:
            inst = 64
        else:
            inst = 128
        triangle_buffer = Buffer(len(indices), int)
        triangle_buffer.copy_from(np.array(indices, dtype=np.int32))
        heap.emplace(mesh_cnt, triangle_buffer)
        accel.add(vertex_buffer, triangle_buffer, visibility_mask=inst)
        mesh_cnt += 1
    # add obj
    # obj_mesh_cnt = mesh_cnt
    # for mesh in obj.mesh_list[0].faces:
    #     indices = [v + len(cornell_box.vertices) for v in mesh]
    #     triangle_buffer = Buffer(len(indices),int)
    #     triangle_buffer.copy_from(np.array(indices, dtype=np.int32))
    #     heap.emplace(obj_mesh_cnt, triangle_buffer)
    #     accel.add(vertex_buffer, triangle_buffer, visibility_mask = 64)
    #     obj_mesh_cnt += 1
    indices = [v + len(cornell_box.vertices) for face in obj.mesh_list[0].faces for v in face]
    triangle_buffer = Buffer(len(indices),int)
    triangle_buffer.copy_from(np.array(indices, dtype=np.int32))
    heap.emplace(mesh_cnt, triangle_buffer)
    accel.add(vertex_buffer, triangle_buffer, visibility_mask = 128)
    accel.update()
    heap.update()

def sample():
    global frame_index, image, accel, res
    raytracing_kernel(image, seed_image, accel, heap, make_int2(
        *res), vertex_buffer, material_buffer, mesh_cnt, frame_index, dispatch_size=(*res, 1))
    accumulate_kernel(accum_image, image, dispatch_size=[*res, 1])
    hdr2ldr_kernel(accum_image, ldr_image, 1.0, dispatch_size=[*res, 1])
    frame_index += 1

if useGUI:
    gui = GUI("Test cornell box", res)


if __name__ == '__main__':
    for i in range(181):
        # 读取obj文件
        print(f"rendering frame {i}....")
        obj = Wavefront(obj_input_folder + f'mpm3d_{i}.obj', collect_faces=True)
        init_buffer(obj)
        init_heap(obj)
        clear_kernel(accum_image, dispatch_size=[*res, 1])
        frame_index = 0
        while True:
            sample()
            if useGUI:
                gui.set_image(ldr_image)
                gui.show()
            if is_to_output and frame_index == 3500:
                ldr_image.to_image(image_output_folder + f"output_{i}.png")
                break
        synchronize()
    if is_to_output:
        # output video
        video = cv2.VideoWriter("mpm3d.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, res)
        for i in range(181):
            video.write(cv2.imread(image_output_folder + f"output_{i}.png"))
        cv2.destroyAllWindows()
        video.release()
