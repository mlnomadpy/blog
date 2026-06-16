import{S as H,F as le,E as Z,r as ce,t as fe,D as m,m as pe,A as d,f as Q,s as _,b as Y,U as de,R as X,p as W,c as J,i as z,g as te,h as xe,j as he}from"./jax.Cg-A9lc1.js";import"./draw.RIl5_ADG.js";const ge=`
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return vec2<u32>(x0, x1);
}`,me=`
const _erf_p: f32 = 0.3275911;
const _erf_a1: f32 = 0.254829592;
const _erf_a2: f32 = -0.284496736;
const _erf_a3: f32 = 1.421413741;
const _erf_a4: f32 = -1.453152027;
const _erf_a5: f32 = 1.061405429;
fn erf(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
fn erfc(x: f32) -> f32 {
  let t = 1.0 / (1.0 + _erf_p * abs(x));
  let P_t = fma(fma(fma(fma(_erf_a5, t, _erf_a4), t, _erf_a3), t, _erf_a2), t, _erf_a1) * t;
  let E = P_t * exp(-x * x);
  return select(2.0 - E, E, x >= 0.0);
}`,V=String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();function A(t,e=!1){switch(t){case m.Bool:return e?"i32":"bool";case m.Int32:return"i32";case m.Uint32:return"u32";case m.Float32:return"f32";case m.Float16:return"f16";default:throw new Error(`Unsupported dtype for WebGPU: ${t}`)}}function ae(t){switch(t){case m.Bool:return"1";case m.Int32:return"2147483647";case m.Uint32:return"4294967295u";case m.Float32:return"inf()";case m.Float16:return"f16(inf())";default:throw new Error(`Unsupported dtype for WebGPU: ${t}`)}}function re(t,e){if(t===m.Bool)return e?"true":"false";if(t===m.Int32)return e.toString();if(t===m.Uint32)return e.toString()+"u";if(t===m.Float32)return Number.isNaN(e)?"nan()":Number.isFinite(e)?"f32("+e.toString()+")":e>0?"inf()":"-inf()";if(t===m.Float16)return Number.isNaN(e)?"f16(nan())":Number.isFinite(e)?"f16("+e.toString()+")":e>0?"f16(inf())":"f16(-inf())";throw new Error(`Unsupported const dtype: ${t}`)}const F=16384;function K(t){let e=t,r=1;return t>65535&&(e=F,r=Math.ceil(t/F)),[e,r]}var _e=class E{static alphaModes=["opaque","premultiplied"];static width=256;static height=256;initialized=!1;deviceStorage;deviceContexts;hostStorage;hostContext;constructor(e){this.device=e}#e(){if(typeof OffscreenCanvas>"u")throw new Error("OffscreenCanvas is not available in this environment, so you cannot read data from WebGPU synchronously. Consider using the async API.");const e=()=>new OffscreenCanvas(E.width,E.height);this.deviceStorage=E.alphaModes.map(e),this.deviceContexts=this.deviceStorage.map((r,s)=>{const i=r.getContext("webgpu");return i.configure({device:this.device,format:"bgra8unorm",usage:GPUTextureUsage.COPY_DST,alphaMode:E.alphaModes[s]}),i}),this.hostStorage=e(),this.hostContext=this.hostStorage.getContext("2d",{willReadFrequently:!0}),this.initialized=!0}read(e,r,s){this.initialized||this.#e();const i=this.deviceStorage,a=this.deviceContexts,p=this.hostContext,g=Math.ceil(s/4),x=E.width*4,b=new ArrayBuffer(g*4);for(let u=0;u<a.length;u++){const v=a[u].getCurrentTexture(),U=(G,R,q)=>{const O=this.device.createCommandEncoder();O.copyBufferToTexture({buffer:e,bytesPerRow:x,offset:q+r},{texture:v},{width:G,height:R,depthOrArrayLayers:1});const N=O.finish();this.device.queue.submit([N]),p.clearRect(0,0,G,R),p.drawImage(i[u],0,0);const L=p.getImageData(0,0,G,R).data,I=new Uint8ClampedArray(b,q,4*G*R),C=E.alphaModes[u];for(let l=0;l<I.length;l+=4)C==="premultiplied"?I[l+3]=L[l+3]:(I[l]=L[l+2],I[l+1]=L[l+1],I[l+2]=L[l])},S=E.width*E.height,w=Math.floor(g/S);let k=g%S;const B=Math.floor(k/E.width);k=k%E.width;let M=0;for(let G=0;G<w;G++)U(E.width,E.height,M),M+=S*4;B>0&&(U(E.width,B,M),M+=B*E.width*4),k>0&&U(k,1,M)}return new Uint8Array(b,0,s)}};function be(t){const e=new Uint32Array(3);return e[0]=t.kind==="sort"?0:1,e[1]=t.mergeStep??0,e[2]=t.mergeStage??0,new Uint8Array(e.buffer)}function ne(t,e,r,s,i){const a=A(e,!0),p=1<<Math.ceil(Math.log2(r||1)),g=Math.ceil(p/2),x=Q(g,t.limits.maxComputeWorkgroupSizeX),b=g/x,u=Math.log2(p),v=Math.min(u,Math.log2(x*2)),U=e===m.Float16,S=z(e)?`${a}(nan())`:ae(e),w=`
${U?"enable f16;":""}
${V}

struct Uniforms {
  kind: u32, // 0 = sort, 1 = merge
  merge_step: u32, // half_block = 2^step
  merge_stage: u32, // only used for merge
}

@group(0) @binding(0) var<storage, read> input: array<${a}>;
@group(0) @binding(1) var<storage, read_write> output: array<${a}>;
${i?"@group(0) @binding(2) var<storage, read_write> output_idx: array<i32>;":""}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

var<workgroup> shared_vals: array<${a}, ${x*2}>;
${i?`var<workgroup> shared_idx: array<i32, ${x*2}>;`:""}

fn compare(a: ${a}, b: ${a}) -> bool {
${z(e)?`
  let min_value = min(a, b);
  return a == min_value && b != min_value;`:"  return a < b;"}
}

fn compare_and_swap(i: u32, j: u32) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
${i?`
  if (
    compare(val_j, val_i) ||
    (!compare(val_i, val_j) && shared_idx[j] < shared_idx[i])
  ) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;
  }`:`
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
  }`}
}

@compute @workgroup_size(${x})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let blockid = wg_id.x + wg_id.y * ${F}u;
  let batch = blockid / ${b}u;
  let wg_in_batch = blockid % ${b}u;

  let tid = local_id.x;
  let base = batch * ${r}u;

  if (uniforms.kind == 0u || (uniforms.kind == 1u && uniforms.merge_step == ${v-1}u)) {
    let wg_base = wg_in_batch * ${x*2}u;

    // Load data into shared memory (2 elements per thread)
    let idx0 = tid * 2u;
    let idx1 = tid * 2u + 1u;
    // Load from input for initial 'sort' pass, then from output (read-write) for 'merge' passes.
    if (uniforms.kind == 0u) {
      shared_vals[idx0] = select(${S}, input[base + wg_base + idx0], wg_base + idx0 < ${r}u);
      shared_vals[idx1] = select(${S}, input[base + wg_base + idx1], wg_base + idx1 < ${r}u);
${i?`
      shared_idx[idx0] = i32(wg_base + idx0);
      shared_idx[idx1] = i32(wg_base + idx1);`:""}
    } else {
      shared_vals[idx0] = select(${S}, output[base + wg_base + idx0], wg_base + idx0 < ${r}u);
      shared_vals[idx1] = select(${S}, output[base + wg_base + idx1], wg_base + idx1 < ${r}u);
${i?`
      shared_idx[idx0] = select(${r}, output_idx[base + wg_base + idx0], wg_base + idx0 < ${r}u);
      shared_idx[idx1] = select(${r}, output_idx[base + wg_base + idx1], wg_base + idx1 < ${r}u);`:""}
    }
    workgroupBarrier();

    let initial_stage = select(0u, ${v-1}u, uniforms.kind != 0u);
    for (var stage = initial_stage; stage < ${v}u; stage++) {
      for (var step1 = stage + 1u; step1 > 0u; step1--) {
        let step = step1 - 1u;
        let half_block = 1u << step;
        let is_first_step = uniforms.kind == 0u && step == stage;

        let block_offset = (tid / half_block) * half_block;
        let local_offset = tid % half_block;
        let i = block_offset * 2u + local_offset;
        let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);
        compare_and_swap(i, j);

        workgroupBarrier();
      }
    }

    if (wg_base + idx0 < ${r}u) {
      output[base + wg_base + idx0] = shared_vals[idx0];
      ${i?"output_idx[base + wg_base + idx0] = shared_idx[idx0];":""}
    }
    if (wg_base + idx1 < ${r}u) {
      output[base + wg_base + idx1] = shared_vals[idx1];
      ${i?"output_idx[base + wg_base + idx1] = shared_idx[idx1];":""}
    }
  } else {
    // Execute single merge pass for a step >= numLocalStages.
    let half_block = 1u << uniforms.merge_step;  // half_block >= workgroupSize * 2
    let thread_in_batch = wg_in_batch * ${x} + tid;
    let is_first_step = uniforms.merge_step == uniforms.merge_stage;

    let block_offset = (thread_in_batch / half_block) * half_block;
    let local_offset = thread_in_batch % half_block;
    let i = block_offset * 2u + local_offset;
    let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);

    // Global version of compare_and_swap()
    if (j < ${r}u) {
      let val_i = output[base + i];
      let val_j = output[base + j];
${i?`
      let idx_i = output_idx[base + i];
      let idx_j = output_idx[base + j];
      if (compare(val_j, val_i) || (!compare(val_i, val_j) && idx_j < idx_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
        output_idx[base + i] = idx_j;
        output_idx[base + j] = idx_i;`:`
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;`}
      }
    }
  }
}
`.trim(),k=K(s*b),B=[{kind:"sort"}];for(let M=v;M<u;M++)for(let G=M;G>=v-1;G--)B.push({kind:"merge",mergeStep:G,mergeStage:M});return[{code:w,numInputs:1,numOutputs:i?2:1,hasUniform:!0,passes:B.map(M=>({grid:k,uniform:be(M)}))}]}function $e(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-1],a=W(s.slice(0,-1));return ne(t,r,i,a,!1)}function we(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-1],a=W(s.slice(0,-1));return ne(t,r,i,a,!0)}function ye(t,e,r){const s=e.inputDtypes[0],i=e.inputShapes[0],a=e.inputShapes[1],p=i[i.length-1],g=a[a.length-2],x=W(i.slice(0,-2)),b=s===m.Float16,u=A(s,!0),v=Q(p,t.limits.maxComputeWorkgroupSizeX),U=`
${b?"enable f16;":""}
${V}

@group(0) @binding(0) var<storage, read> a: array<${u}>;
@group(0) @binding(1) var<storage, read> b: array<${u}>;
@group(0) @binding(2) var<storage, read_write> x: array<${u}>;

// Shared memory for the current pivot value x[j]
var<workgroup> x_j: ${u};

@compute @workgroup_size(${v})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_idx = wg_id.x + wg_id.y * ${F}u;
  let mat_idx = wg_idx / ${g}u;
  let rhs_idx = wg_idx % ${g}u;

  if (mat_idx >= ${x}u) {
    return;
  }

  let a_base = mat_idx * ${p*p}u;
  let bx_base = (mat_idx * ${g}u + rhs_idx) * ${p}u;
  let tid = local_id.x;

  // Step 1: Copy b to x (threads collaborate)
  for (var idx = tid; idx < ${p}u; idx += ${v}u) {
    x[bx_base + idx] = b[bx_base + idx];
  }
  storageBarrier();

  // Step 2: Back-substitution from j = n-1 down to 0
  for (var jj = 0u; jj < ${p}u; jj++) {
    let j = ${p-1}u - jj;

    // Thread 0 computes x[j] = x[j] / a[j,j]
    if (tid == 0u) {
      ${r.unitDiagonal?"x_j = x[bx_base + j];":`x_j = x[bx_base + j] / a[a_base + j * ${p}u + j];`}
      x[bx_base + j] = x_j;
    }
    workgroupBarrier();  // Sync shared memory x_j

    // All threads subtract x[j] * a[i,j] from x[i] for i < j
    for (var i = tid; i < j; i += ${v}u) {
      x[bx_base + i] -= x_j * a[a_base + i * ${p}u + j];
    }
    workgroupBarrier();
    storageBarrier();
  }
}
`.trim(),S=x*g,w=K(S);return[{code:U,numInputs:2,numOutputs:1,hasUniform:!1,passes:[{grid:w}]}]}function ve(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-1],a=W(s.slice(0,-2)),p=r===m.Float16,g=A(r,!0),x=Q(i,t.limits.maxComputeWorkgroupSizeX),b=`
${p?"enable f16;":""}
${V}

@group(0) @binding(0) var<storage, read> input: array<${g}>;
@group(0) @binding(1) var<storage, read_write> output: array<${g}>;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${g};

@compute @workgroup_size(${x})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${F}u;
  if (batch >= ${a}u) {
    return;
  }

  let base = batch * ${i*i}u;
  let tid = local_id.x;

  // Zero out output and copy lower triangle from input (threads collaborate)
  for (var idx = tid; idx < ${i*i}u; idx += ${x}u) {
    let row = idx / ${i}u;
    let col = idx % ${i}u;
    output[base + idx] = select(0, input[base + idx], col <= row);
  }
  storageBarrier();

  // Cholesky-Crout algorithm: process column by column
  for (var j = 0u; j < ${i}u; j++) {
    // Step 1: All threads compute sum for their rows i >= j in parallel
    // sum = A[i][j] - sum(L[i][k] * L[j][k] for k < j)
    for (var i = j + tid; i < ${i}u; i += ${x}u) {
      var sum = output[base + i * ${i}u + j];
      for (var k = 0u; k < j; k++) {
        sum -= output[base + i * ${i}u + k] * output[base + j * ${i}u + k];
      }
      output[base + i * ${i}u + j] = sum;
    }
    storageBarrier();

    // Step 2: Thread 0 computes L[j][j] = sqrt(output[j][j])
    if (tid == 0u) {
      L_jj = sqrt(output[base + j * ${i}u + j]);
      output[base + j * ${i}u + j] = L_jj;
    }
    workgroupBarrier();

    // Step 3: All threads divide output[i][j] by L[j][j] for i > j
    for (var i = j + 1u + tid; i < ${i}u; i += ${x}u) {
      output[base + i * ${i}u + j] /= L_jj;
    }
    storageBarrier();
  }
}
`.trim(),u=K(a);return[{code:b,numInputs:1,numOutputs:1,hasUniform:!1,passes:[{grid:u}]}]}function Se(t,e){const r=e.inputDtypes[0],s=e.inputShapes[0],i=s[s.length-2],a=s[s.length-1],p=Math.min(i,a),g=W(s.slice(0,-2)),x=r===m.Float16,b=A(r,!0),u=Q(Math.max(i,a),t.limits.maxComputeWorkgroupSizeX),v=`
${x?"enable f16;":""}
${V}

@group(0) @binding(0) var<storage, read> input: array<${b}>;
@group(0) @binding(1) var<storage, read_write> lu: array<${b}>;
@group(0) @binding(2) var<storage, read_write> pivots: array<i32>;
@group(0) @binding(3) var<storage, read_write> perm: array<i32>;

var<workgroup> pivot_row: u32;
var<workgroup> pivot_val: ${b};

@compute @workgroup_size(${u})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${F}u;
  if (batch >= ${g}u) {
    return;
  }

  let lu_base = batch * ${i*a}u;
  let piv_base = batch * ${p}u;
  let perm_base = batch * ${i}u;
  let tid = local_id.x;

  // Copy input to lu
  for (var idx = tid; idx < ${i*a}u; idx += ${u}u) {
    lu[lu_base + idx] = input[lu_base + idx];
  }
  // Initialize permutation
  for (var idx = tid; idx < ${i}u; idx += ${u}u) {
    perm[perm_base + idx] = i32(idx);
  }
  storageBarrier();

  // LU decomposition with partial pivoting
  for (var j = 0u; j < ${p}u; j++) {
    // Step 1: Thread 0 finds pivot (max abs value in column j, rows >= j)
    if (tid == 0u) {
      var max_val = abs(lu[lu_base + j * ${a}u + j]);
      var max_row = j;
      for (var i = j + 1u; i < ${i}u; i++) {
        let val = abs(lu[lu_base + i * ${a}u + j]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }
      pivot_row = max_row;
      pivot_val = lu[lu_base + max_row * ${a}u + j];
      pivots[piv_base + j] = i32(max_row);
    }
    workgroupBarrier();

    // Step 2: Swap rows j and pivot_row (threads collaborate)
    let pr = pivot_row;
    if (pr != j) {
      for (var col = tid; col < ${a}u; col += ${u}u) {
        let tmp = lu[lu_base + j * ${a}u + col];
        lu[lu_base + j * ${a}u + col] = lu[lu_base + pr * ${a}u + col];
        lu[lu_base + pr * ${a}u + col] = tmp;
      }
      if (tid == 0u) {
        let tmp_p = perm[perm_base + j];
        perm[perm_base + j] = perm[perm_base + pr];
        perm[perm_base + pr] = tmp_p;
      }
    }
    storageBarrier();

    // Step 3: Compute L[i][j] and update submatrix
    // Each thread handles one row i > j
    for (var i = j + 1u + tid; i < ${i}u; i += ${u}u) {
      let factor = lu[lu_base + i * ${a}u + j] / pivot_val;
      lu[lu_base + i * ${a}u + j] = factor; // L[i][j]
      for (var k = j + 1u; k < ${a}u; k++) {
        lu[lu_base + i * ${a}u + k] -= factor * lu[lu_base + j * ${a}u + k];
      }
    }
    storageBarrier();
  }
}
`.trim(),U=K(g);return[{code:v,numInputs:1,numOutputs:3,hasUniform:!1,passes:[{grid:U}]}]}function ie(t,e){switch(e.name){case X.Sort:return $e(t,e.type);case X.Argsort:return we(t,e.type);case X.TriangularSolve:return ye(t,e.type,e.params);case X.Cholesky:return ve(t,e.type);case X.LU:return Se(t,e.type);default:throw new de(e.name,"webgpu")}}const ee=4096,se=new WeakMap;function je(t){return{querySet:t.createQuerySet({type:"timestamp",count:ee}),resolve:t.createBuffer({size:ee*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),dst:t.createBuffer({size:ee*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),nextIndex:0,entries:[]}}function ke(t,e,r,s,i){const a=xe(r);a.properties.push(["passes",`${s}`]),a.properties.push(["source",i]),e.batch.entries.push({...a,beginIndex:e.beginIndex,endIndex:e.endIndex}),Be(t)}function Be(t){queueMicrotask(()=>{const e=se.get(t);e&&e.entries.length>0&&(Ue(t,e),se.set(t,je(t)))})}function Ue(t,e){if(e.entries.length===0)return;const r=e.nextIndex,s=t.createCommandEncoder();s.resolveQuerySet(e.querySet,0,r,e.resolve,0),s.copyBufferToBuffer(e.resolve,0,e.dst,0,r*8),t.queue.submit([s.finish()]);const{entries:i}=e;e.dst.mapAsync(GPUMapMode.READ).then(()=>{try{const a=new BigInt64Array(e.dst.getMappedRange()),p=a[i[i.length-1].endIndex],g=performance.now();for(const x of i){const b=g+Number(a[x.beginIndex]-p)/1e6,u=g+Number(a[x.endIndex]-p)/1e6;he("webgpu",x,b,u)}}finally{e.dst.unmap(),e.querySet.destroy(),e.resolve.destroy(),e.dst.destroy()}})}var Ie=class{type="webgpu";maxArgs;pipelines;syncReader;buffers;nextSlot;#e=new Map;#t;constructor(t){this.device=t,this.maxArgs=this.device.limits.maxStorageBuffersPerShaderStage-1,this.pipelines=new Ee(t),this.syncReader=new _e(t),this.buffers=new Map,this.nextSlot=1,this.#t=this.#r(4),t.addEventListener("uncapturederror",e=>{console.error("Uncaptured error in WebGPU backend:",e.error.message)})}malloc(t,e){let r;const s=Math.ceil(t/4)*4;if(t===0)r=this.#t;else if(e){if(e.byteLength!==t)throw new Error("initialData size does not match buffer size");if(e.byteLength<4096)r=this.#r(s,{mapped:!0}),new Uint8Array(r.getMappedRange(),0,t).set(e),r.unmap();else if(r=this.#r(s),e.byteLength%4===0)this.device.queue.writeBuffer(r,0,e);else{const a=e.byteLength-e.byteLength%4;this.device.queue.writeBuffer(r,0,e,0,a);const p=new Uint8Array(4);p.set(e.subarray(a)),this.device.queue.writeBuffer(r,a,p)}}else r=this.#r(s);const i=this.nextSlot++;return this.buffers.set(i,{buffer:r,size:t,ref:1}),i}incRef(t){const e=this.buffers.get(t);if(!e)throw new H(t);e.ref++}decRef(t){const e=this.buffers.get(t);if(!e)throw new H(t);e.ref--,e.ref===0&&(this.buffers.delete(t),e.buffer!==this.#t&&e.buffer.destroy())}async read(t,e,r){const{buffer:s,size:i}=this.#i(t);if(s===this.#t)return new Uint8Array;e===void 0&&(e=0),r===void 0&&(r=i-e);const a=Math.ceil(r/4)*4,p=this.#r(a,{read:!0});try{const g=this.device.createCommandEncoder();g.copyBufferToBuffer(s,e,p,0,a),this.device.queue.submit([g.finish()]),await p.mapAsync(GPUMapMode.READ);const x=p.getMappedRange();return new Uint8Array(x.slice(),0,r)}finally{p.destroy()}}readSync(t,e,r){const{buffer:s,size:i}=this.#i(t);return s===this.#t?new Uint8Array:(e===void 0&&(e=0),r===void 0&&(r=i-e),this.syncReader.read(s,e,r))}#s(t){const e=le.hash(t);let r=this.#e.get(e);return r||(r=Me(this.device,t),this.#e.set(e,r)),r}async prepareKernel(t){const e=this.#s(t),r=await this.pipelines.prepare(e);return new Z(t,[{...e,pipeline:r}])}prepareKernelSync(t){const e=this.#s(t),r=this.pipelines.prepareSync(e);return new Z(t,[{...e,pipeline:r}])}async prepareRoutine(t){const e=ie(this.device,t),r=await Promise.all(e.map(async s=>{const i=await this.pipelines.prepare(s);return{...s,pipeline:i}}));return new Z(t,r)}prepareRoutineSync(t){const r=ie(this.device,t).map(s=>{const i=this.pipelines.prepareSync(s);return{...s,pipeline:i}});return new Z(t,r)}dispatch(t,e,r){const s=e.map(a=>this.#i(a).buffer),i=r.map(a=>this.#i(a).buffer);Pe(this.device,t,s,i)}#i(t){const e=this.buffers.get(t);if(!e)throw new H(t);return{buffer:e.buffer,size:e.size}}#r(t,{mapped:e=!1,read:r=!1}={}){if(r&&e)throw new Error("mapped and read cannot both be true");return this.device.createBuffer({size:t,usage:r?GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,mappedAtCreation:e})}};function Me(t,e){const r=fe(e),{nargs:s,reduction:i}=e,a=Array.from({length:s},(l,c)=>`in${c}`),p=[];let g="";const x=Symbol("pushIndent"),b=Symbol("popIndent"),u=(...l)=>{for(const c of l)c===x?g+="  ":c===b?g=g.slice(0,-2):p.push(c&&g+c)};if(r.exp.some(l=>l.dtype===m.Float16)||r.epilogue?.some(l=>l.dtype===m.Float16)){if(!t.features.has("shader-f16"))throw new Error("WebGPU device does not support shader-f16 feature");u("enable f16;")}u(V);const v=pe(r.exp.distinctOps(),r.epilogue?.distinctOps());v.has(d.Threefry2x32)&&u(ge),(v.has(d.Erf)||v.has(d.Erfc))&&u(me),u("");const U=Array.from({length:s},()=>null);r.exp.fold(l=>{l.op===d.GlobalIndex&&(U[l.arg[0]]=l.dtype)}),r.epilogue?.fold(l=>{l.op===d.GlobalIndex&&(U[l.arg[0]]=l.dtype)});for(let l=0;l<s;l++){const c=A(U[l]??m.Float32,!0);u(`@group(0) @binding(${l}) var<storage, read> ${a[l]} : array<${c}>;`)}const S=A(e.dtype,!0);u(`@group(0) @binding(${s}) var<storage, read_write> result : array<${S}>;`);const w=Q(r.threadCount,256),k=Math.ceil(r.threadCount/w),[B,M]=K(k);if(u("",`@compute @workgroup_size(${w})`,"fn main(@builtin(global_invocation_id) id : vec3<u32>) {",x),M===1)u(`if (id.x >= ${r.threadCount}) { return; }`,"let gidx: i32 = i32(id.x);");else{const l=B*w;u(`if (${l} * id.y + id.x >= ${r.threadCount}) { return; }`,`let gidx: i32 = i32(${l} * id.y + id.x);`)}let G=0;const R=()=>`alu${G++}`,q=l=>l.match(/^alu[0-9]+$/);a.length>0&&u(a.map(l=>`_ = &${l};`).join(" "));const O=new Map,N=new Set,L=l=>{if(O.set(l,(O.get(l)??0)+1),!N.has(l)){N.add(l);for(const c of l.src)L(c)}},I=new Map,C=l=>{if(I.has(l))return I.get(l);const{op:c,src:$,dtype:y,arg:P}=l;let f="";if(J.Binary.has(c)||J.Compare.has(c)){const n=C($[0]),o=C($[1]);if(c===d.Add)y===m.Bool?f=`(${n} || ${o})`:f=`(${n} + ${o})`;else if(c===d.Sub)f=`(${n} - ${o})`;else if(c===d.Mul)y===m.Bool?f=`(${n} && ${o})`:f=`(${n} * ${o})`;else if(c===d.Idiv)f=z(y)?`trunc(${n} / ${o})`:`(${n} / ${o})`;else if(c===d.Mod)f=`(${n} % ${o})`;else if(c===d.Min)y===m.Bool?f=`(${n} && ${o})`:f=`min(${_(n)}, ${_(o)})`;else if(c===d.Max)y===m.Bool?f=`(${n} || ${o})`:f=`max(${_(n)}, ${_(o)})`;else if(c===d.BitCombine)P==="and"?f=`(${n} & ${o})`:P==="or"?f=`(${n} | ${o})`:f=y===m.Bool?`(${n} != ${o})`:`(${n} ^ ${o})`;else if(c===d.BitShift)P==="shl"?f=`(${n} << ${o})`:f=`(${n} >> ${o})`;else if(c===d.Cmplt)f=`(${n} < ${o})`;else if(c===d.Cmpne)if(z($[0].dtype)){const h=q(n)?n:R();h!==n&&u(`let ${h} = ${n};`),f=`(${h} != ${o} || min(${h}, ${A($[0].dtype)}(inf())) != ${h})`}else f=`(${n} != ${o})`}else if(J.Unary.has(c))if(c===d.Reciprocal&&$[0].op===d.Sqrt)f=`inverseSqrt(${C($[0].src[0])})`;else{const n=C($[0]);if(c===d.Sin)f=`sin(${_(n)})`;else if(c===d.Cos)f=`cos(${_(n)})`;else if(c===d.Asin)f=`asin(${_(n)})`;else if(c===d.Atan)f=`atan(${_(n)})`;else if(c===d.Exp)f=`exp(${_(n)})`;else if(c===d.Log)f=`log(${_(n)})`;else if(c===d.Erf||c===d.Erfc){const o=c===d.Erf?"erf":"erfc";y!==m.Float32?f=`${A(y)}(${o}(f32(${_(n)})))`:f=`${o}(${_(n)})`}else if(c===d.Sqrt)f=`sqrt(${_(n)})`;else if(c===d.Reciprocal)f=`(1.0 / ${n})`;else if(c===d.Floor)f=`floor(${_(n)})`;else if(c===d.Ceil)f=`ceil(${_(n)})`;else if(c===d.Cast){const o=A($[0].dtype),h=A(y);if(z($[0].dtype)&&!(z(y)||y===m.Bool)){const j=ae(y),T=q(n)?n:R();T!==n&&u(`let ${T}: ${o} = ${_(n)};`),f=`select(${h}(${T}), ${j}, ${T} >= ${o}(${j}))`}else f=`${h}(${_(n)})`}else c===d.Bitcast&&(f=`bitcast<${A(y)}>(${_(n)})`)}else if(c===d.Where)f=`select(${_(C($[2]))}, ${_(C($[1]))}, ${_(C($[0]))})`;else if(c===d.Threefry2x32){const n=R(),[o,h,j,T]=$.map(ue=>_(C(ue)));if(u(`let ${n} = threefry2x32(vec2(${o}, ${h}), vec2(${j}, ${T}));`),P==="xor")f=`(${n}.x ^ ${n}.y)`;else if(P===0)f=`${n}.x`;else if(P===1)f=`${n}.y`;else throw new te(c,y,"webgpu",P)}else{if(c===d.Const)return re(y,P);if(c===d.Special)return P[0];if(c===d.Variable)return P;c===d.GlobalIndex&&(f=`${a[P[0]]}[${_(C($[0]))}]`,y===m.Bool&&(f=`(${f} != 0)`))}if(!f)throw new te(c,y,"webgpu",P);const D=A(y);if((O.get(l)??0)>1){const n=R();return I.set(l,n),u(`let ${n}: ${D} = ${_(f)};`),n}else return I.set(l,f),f};if(i){if((r.size.groups??1)>1)throw new Error("WebGPU backend does not support group optimization yet");const l=r.size.unroll??1,c=r.size.upcast??1,$=[...Array(c)].map((o,h)=>`acc${h}`);for(let o=0;o<c;o++)u(`var ${$[o]}: ${A(i.dtype)} = ${re(i.dtype,i.identity)};`);u(`for (var ridx: i32 = 0; ridx < ${r.size.reduce}; ridx++) {`,x);const y=[],P=new Map;for(let o=0;o<c;o++){y.push([]);for(let h=0;h<l;h++){const j=r.exp.substitute({upcast:Y.i32(o),unroll:Y.i32(h)});y[o].push(j.simplify(P)),L(y[o][h])}}const f=y.map(o=>o.map(C).map(_));for(let o=0;o<c;o++){let h=f[o][0];for(let j=1;j<l;j++)if(i.op===d.Add)h=`${h} + ${f[o][j]}`;else if(i.op===d.Mul)h=`${h} * ${f[o][j]}`;else if(i.op===d.Min)h=i.dtype===m.Bool?`(${h} && ${f[o][j]})`:`min(${h}, ${f[o][j]})`;else if(i.op===d.Max)h=i.dtype===m.Bool?`(${h} || ${f[o][j]})`:`max(${h}, ${f[o][j]})`;else throw new Error(`Unsupported reduction op: ${i.op}`);if(i.op===d.Add)u(`${$[o]} += ${h};`);else if(i.op===d.Mul)u(`${$[o]} *= ${h};`);else if(i.op===d.Min)i.dtype===m.Bool?u(`${$[o]} = ${$[o]} && ${h};`):u(`${$[o]} = min(${$[o]}, ${h});`);else if(i.op===d.Max)i.dtype===m.Bool?u(`${$[o]} = ${$[o]} || ${h};`):u(`${$[o]} = max(${$[o]}, ${h});`);else throw new Error(`Unsupported reduction op: ${i.op}`)}u(b,"}"),I.clear(),O.clear(),N.clear();const D=[],n=[];for(let o=0;o<c;o++){const h=r.outputIdxExp.substitute({upcast:Y.i32(o)});D.push(h.simplify(P)),L(D[o]),n.push(r.epilogue.substitute({acc:Y.variable(i.dtype,$[o]),upcast:Y.i32(o)}).simplify(P)),L(n[o])}for(let o=0;o<c;o++){const h=_(C(D[o]));let j=_(C(n[o]));S!==A(n[o].dtype)&&(j=`${S}(${j})`),u(`result[${h}] = ${j};`)}}else{L(r.exp);let l=_(C(r.exp));S!==A(r.exp.dtype)&&(l=`${S}(${l})`),u(`result[gidx] = ${l};`)}return u(b,"}"),{code:p.join(`
`),numInputs:s,numOutputs:1,hasUniform:!1,passes:[{grid:[B,M]}]}}function Pe(t,e,r,s){const{data:i,source:a}=e,p=t.createCommandEncoder();for(const{pipeline:g,...x}of i){if(r.length!==x.numInputs||s.length!==x.numOutputs)throw new Error(`webgpu: expected ${x.numInputs} inputs and ${x.numOutputs} outputs, got ${r.length} inputs and ${s.length} outputs`);const b=x.passes.filter(({grid:w})=>W(w)>0);if(b.length===0)continue;const u=void 0,v=t.createBindGroup({layout:g.getBindGroupLayout(0),entries:[...r.map((w,k)=>({binding:k,resource:{buffer:w}})),...s.map((w,k)=>({binding:r.length+k,resource:{buffer:w}}))]});let U=null,S=0;if(x.hasUniform){const w=b.map(({uniform:M})=>M),[k,B]=Ce(t,w);S=B,U=t.createBindGroup({layout:g.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:k,size:B}}]})}for(let w=0;w<b.length;w++){const{grid:k}=b[w],B=p.beginComputePass({timestampWrites:u?{querySet:u.batch.querySet,beginningOfPassWriteIndex:w===0?u.beginIndex:void 0,endOfPassWriteIndex:w===b.length-1?u.endIndex:void 0}:void 0});B.setPipeline(g),B.setBindGroup(0,v),U&&B.setBindGroup(1,U,[w*S]),B.dispatchWorkgroups(k[0],k[1]),B.end()}u&&ke(t,u,a,b.length,x.code)}t.queue.submit([p.finish()])}function Ce(t,e){for(const p of e)if(!p||p.byteLength===0||p.byteLength!==e[0].byteLength)throw new Error("webgpu: Uniform mismatch between shader passes");const r=t.limits.minUniformBufferOffsetAlignment,s=Math.ceil(e[0].byteLength/r)*r,i=t.createBuffer({size:s*e.length,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:!0}),a=new Uint8Array(i.getMappedRange());for(let p=0;p<e.length;p++)a.set(e[p],p*s);return i.unmap(),[i,s]}var Ee=class{cache;inProgress;constructor(t){this.device=t,this.cache=new Map,this.inProgress=new Map}#e(t){if(t.numInputs+t.numOutputs>this.device.limits.maxStorageBuffersPerShaderStage){const r=t.numInputs+t.numOutputs,s=this.device.limits.maxStorageBuffersPerShaderStage;throw new Error(`Too many buffers (${r}) for WebGPU pipeline (max: ${s})`)}const e=[this.device.createBindGroupLayout({entries:ce(t.numInputs+t.numOutputs).map(r=>({binding:r,visibility:GPUShaderStage.COMPUTE,buffer:{type:r<t.numInputs?"read-only-storage":"storage"}}))})];return t.hasUniform&&e.push(this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0}}]})),this.device.createPipelineLayout({bindGroupLayouts:e})}async prepare(t){const e=this.cache.get(t.code);if(e)return e;const r=this.inProgress.get(t.code);if(r)return await r;const s=this.device.createShaderModule({code:t.code}),i=(async()=>{this.device.pushErrorScope("validation");try{const p=await this.device.createComputePipelineAsync({layout:this.#e(t),compute:{module:s,entryPoint:"main"}});return await this.device.popErrorScope(),p}catch{const g=await this.device.popErrorScope(),x=await oe(s,g,t.code);throw new Error(x)}})();this.inProgress.set(t.code,i);const a=await i;return this.cache.set(t.code,a),a}prepareSync(t){const e=this.cache.get(t.code);if(e)return e;const r=this.device.createShaderModule({code:t.code});this.device.pushErrorScope("validation");const s=this.device.createComputePipeline({layout:this.#e(t),compute:{module:r,entryPoint:"main"}});return this.device.popErrorScope().then(async i=>{if(i!==null){const a=await oe(r,i,t.code);console.error(a)}}),this.cache.set(t.code,s),s}};async function oe(t,e,r){let s=`Failed to compile shader: ${e?e.message:"(no error scope)"}`;const i=await t.getCompilationInfo();for(const a of i.messages)s+=`
  [${a.type} at ${a.lineNum}:${a.linePos}] ${a.message}`;return r&&(s+=`

${r}`),s}export{Ie as WebGPUBackend};
