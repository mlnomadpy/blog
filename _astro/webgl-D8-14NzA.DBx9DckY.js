import{S as R,E as y,U as S,l as D,D as E,s as d,A as c,r as P,i as g,g as B,h as O}from"./jax.DE6Pan_r.js";import"./draw.B8cP0lzX.js";const N=`
uvec2 threefry2x32(uvec2 key, uvec2 ctr) {
  uint ks0 = key.x;
  uint ks1 = key.y;
  uint ks2 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  uint x0 = ctr.x + ks0;
  uint x1 = ctr.y + ks1;

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

  return uvec2(x0, x1);
}`,v=`
const float _erf_p = 0.3275911;
const float _erf_a1 = 0.254829592;
const float _erf_a2 = -0.284496736;
const float _erf_a3 = 1.421413741;
const float _erf_a4 = -1.453152027;
const float _erf_a5 = 1.061405429;
float erf(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  return sign(x) * (1.0 - P_t * exp(-x * x));
}
float erfc(float x) {
  float t = 1.0 / (1.0 + _erf_p * abs(x));
  float P_t = (((((_erf_a5 * t) + _erf_a4) * t + _erf_a3) * t + _erf_a2) * t + _erf_a1) * t;
  float E = P_t * exp(-x * x);
  return x >= 0.0 ? E : 2.0 - E;
}`;var H=class{type="webgl";maxArgs=8;gl;#t;#e;#r;#o;constructor(t){this.gl=t,this.#t=t.createFramebuffer(),this.#e=new Map,this.#r=new Map,this.#o=1}malloc(t,o){const n=this.gl,e=Math.ceil(t/4)||1,r=Math.ceil(e/4)||1,{width:$,height:a}=L(r),u=n.createTexture();if(!u)throw new Error("Failed to create texture");n.bindTexture(n.TEXTURE_2D,u),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MIN_FILTER,n.NEAREST),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_MAG_FILTER,n.NEAREST),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE),n.texParameteri(n.TEXTURE_2D,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE);const x=$*a*4;let l=null;o&&(l=new Float32Array(x),new Uint8Array(l.buffer).set(o)),n.texImage2D(n.TEXTURE_2D,0,n.RGBA32F,$,a,0,n.RGBA,n.FLOAT,l),n.bindTexture(n.TEXTURE_2D,null);const i=this.#o++;return this.#e.set(i,{ref:1,size:t,texture:u,width:$,height:a}),i}incRef(t){const o=this.#e.get(t);if(!o)throw new R(t);o.ref++}decRef(t){const o=this.#e.get(t);if(!o)throw new R(t);o.ref--,o.ref===0&&(this.gl.deleteTexture(o.texture),this.#e.delete(t))}async read(t,o,n){const e=this.#e.get(t);if(!e)throw new R(t);const r=this.gl;o===void 0&&(o=0),n===void 0&&(n=e.size-o),r.bindFramebuffer(r.FRAMEBUFFER,this.#t),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,e.texture,0);const $=e.width*e.height*4*4,a=new Float32Array($/4),u=r.createBuffer();if(!u)throw new Error("Failed to create PBO");r.bindBuffer(r.PIXEL_PACK_BUFFER,u),r.bufferData(r.PIXEL_PACK_BUFFER,$,r.STREAM_READ),r.readPixels(0,0,e.width,e.height,r.RGBA,r.FLOAT,0);const x=r.getError();if(x!==r.NO_ERROR)throw r.deleteBuffer(u),new Error(`WebGL error after readPixels: ${x}`);const l=r.fenceSync(r.SYNC_GPU_COMMANDS_COMPLETE,0);if(!l)throw new Error("Failed to create sync object");r.flush(),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),r.bindFramebuffer(r.FRAMEBUFFER,null),await new Promise((s,f)=>{const _=()=>{const h=r.clientWaitSync(l,0,0);if(h===r.TIMEOUT_EXPIRED){setTimeout(_,5);return}if(h===r.WAIT_FAILED){r.deleteSync(l),r.deleteBuffer(u),f(new Error("clientWaitSync failed"));return}s()};_()}),r.deleteSync(l),r.bindBuffer(r.PIXEL_PACK_BUFFER,u),r.getBufferSubData(r.PIXEL_PACK_BUFFER,0,a),r.bindBuffer(r.PIXEL_PACK_BUFFER,null),r.deleteBuffer(u);const i=new Uint8Array(a.buffer);return new Uint8Array(i.slice(o,o+n))}readSync(t,o,n){const e=this.#e.get(t);if(!e)throw new R(t);const r=this.gl;o===void 0&&(o=0),n===void 0&&(n=e.size-o),r.bindFramebuffer(r.FRAMEBUFFER,this.#t),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,e.texture,0);const $=e.width*e.height*4,a=new Float32Array($);r.readPixels(0,0,e.width,e.height,r.RGBA,r.FLOAT,a),r.bindFramebuffer(r.FRAMEBUFFER,null);const u=new Uint8Array(a.buffer);return new Uint8Array(u.slice(o,o+n))}async prepareKernel(t){return this.prepareKernelSync(t)}prepareKernelSync(t){const o=C(t),n=this.#r.get(o.code);if(n)return new y(t,n);const e=k(this.gl,o);return this.#r.set(o.code,e),new y(t,e)}prepareRoutine(t){throw new S(t.name,"webgl")}prepareRoutineSync(t){throw new S(t.name,"webgl")}dispatch(t,o,n){const e=this.gl;if(e.isContextLost())throw new Error("WebGL context lost - cannot dispatch");const{program:r,inputLocations:$}=t.data;if(o.length!==t.data.numInputs)throw new Error(`Expected ${t.data.numInputs} inputs, got ${o.length}`);if(n.length!==1)throw new Error(`Expected 1 output, got ${n.length}`);const a=this.#e.get(n[0]);if(!a)throw new R(n[0]);e.bindFramebuffer(e.FRAMEBUFFER,this.#t),e.framebufferTexture2D(e.FRAMEBUFFER,e.COLOR_ATTACHMENT0,e.TEXTURE_2D,a.texture,0);const u=e.checkFramebufferStatus(e.FRAMEBUFFER);if(u!==e.FRAMEBUFFER_COMPLETE)throw new Error(`Framebuffer incomplete: ${u}`);e.viewport(0,0,a.width,a.height),e.useProgram(r);for(let l=0;l<o.length;l++){const i=this.#e.get(o[l]);if(!i)throw new R(o[l]);e.activeTexture(e.TEXTURE0+l),e.bindTexture(e.TEXTURE_2D,i.texture),$[l]!==null&&e.uniform1i($[l],l)}e.drawArrays(e.TRIANGLES,0,3);const x=e.getError();if(x!==e.NO_ERROR){let l;throw x===e.INVALID_ENUM?l="INVALID_ENUM":x===e.INVALID_VALUE?l="INVALID_VALUE":x===e.INVALID_OPERATION?l="INVALID_OPERATION":x===e.INVALID_FRAMEBUFFER_OPERATION?l="INVALID_FRAMEBUFFER_OPERATION":x===e.OUT_OF_MEMORY?l="OUT_OF_MEMORY":x===e.CONTEXT_LOST_WEBGL?l="CONTEXT_LOST_WEBGL":l=`UNKNOWN(${x})`,new Error(`WebGL error after drawArrays: ${l}`)}e.bindFramebuffer(e.FRAMEBUFFER,null),e.useProgram(null)}};function C(t){const o=D(t),{nargs:n,reduction:e}=t,r=t.dtype,$=Math.ceil(t.size/4)||1,a=L($),u=Array(n).fill(E.Float32),x={erf:!1,threefry:!1},l=p=>{p.op===c.GlobalIndex?u[p.arg[0]]=p.dtype:p.op===c.Erf||p.op===c.Erfc?x.erf=!0:p.op===c.Threefry2x32&&(x.threefry=!0)};o.exp.fold(l),o.epilogue?.fold(l);const i=[];let s="";const f=Symbol("pushIndent"),_=Symbol("popIndent"),h=(...p)=>{for(const F of p)F===f?s+="  ":F===_?s=s.slice(0,-2):i.push(F&&s+F)};h("#version 300 es","precision highp float;","precision highp int;","");const m=Array.from({length:n},(p,F)=>`in${F}`),T=U(r);for(let p=0;p<n;p++)h(`uniform highp sampler2D ${m[p]};`);h("out vec4 out0;");const A=new Set;for(const p of u)A.add(p);for(const p of A)h(V(p));if(x.erf&&h(v),x.threefry&&h(N),h(`${T} compute(int gidx) {`,f,`${T} result = ${I(r,0)};`,`if (gidx < ${t.size}) {`,f),e){const p=U(e.dtype),F=I(e.dtype,e.identity);h(`${p} acc = ${F};`,`for (int ridx = 0; ridx < ${o.size.reduce}; ridx++) {`,f);const w=b(o.exp,m,u);if(e.op===c.Add)h(`acc += ${d(w)};`);else if(e.op===c.Mul)h(`acc *= ${d(w)};`);else if(e.op===c.Min)e.dtype!==E.Bool?h(`acc = min(acc, ${d(w)});`):h(`acc = acc && ${w};`);else if(e.op===c.Max)e.dtype!==E.Bool?h(`acc = max(acc, ${d(w)});`):h(`acc = acc || ${w};`);else throw new Error(`Unsupported reduction op: ${e.op}`);h(_,"}"),h(`result = ${b(o.epilogue,m,u)};`)}else{const p=b(o.exp,m,u);h(`result = ${d(p)};`)}return h(_,"}","return result;",_,`}
`),h("void main() {",f,"ivec2 fragCoord = ivec2(gl_FragCoord.xy);",`int texelIdx = fragCoord.y * ${a.width} + fragCoord.x;`,`${T} result0 = compute(texelIdx * 4);`,`${T} result1 = compute(texelIdx * 4 + 1);`,`${T} result2 = compute(texelIdx * 4 + 2);`,`${T} result3 = compute(texelIdx * 4 + 3);`,`out0 = vec4(${P(4).map(p=>W(r,`result${p}`)).join(", ")});`),h(_,"}"),{code:i.join(`
`),numInputs:n,outputSize:[a.width,a.height],outputDtype:r}}function M(t,o,n){const e=t.createShader(o);if(t.shaderSource(e,n),t.compileShader(e),!t.getShaderParameter(e,t.COMPILE_STATUS))throw new Error(t.getShaderInfoLog(e)??"Unknown shader compile error");return e}function X(t,o,n){const e=t.createProgram();if(t.attachShader(e,M(t,t.VERTEX_SHADER,o)),t.attachShader(e,M(t,t.FRAGMENT_SHADER,n)),t.linkProgram(e),!t.getProgramParameter(e,t.LINK_STATUS))throw new Error(t.getProgramInfoLog(e)??"Unknown program link error");return e}const G=`#version 300 es
precision highp float;
const vec2 pos[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0); }
`;function k(t,o){const n=X(t,G,o.code),e=[];for(let r=0;r<o.numInputs;r++)e.push(t.getUniformLocation(n,`in${r}`));return{...o,program:n,inputLocations:e}}function L(t){let n=Math.min(Math.ceil(Math.sqrt(t)),16384);n=Math.min(1<<Math.ceil(Math.log2(n)),16384);const e=Math.min(Math.ceil(t/n),16384);return{width:n,height:e}}function U(t){switch(t){case E.Float32:return"float";case E.Int32:return"int";case E.Uint32:return"uint";case E.Bool:return"bool";default:throw new Error(`Unsupported dtype for WebGL: ${t}`)}}function V(t){const o=`load_${t}`,n=U(t);let e;if(g(t))e="val";else if(t===E.Int32)e="floatBitsToInt(val)";else if(t===E.Uint32)e="floatBitsToUint(val)";else if(t===E.Bool)e="floatBitsToInt(val) != 0";else throw new Error(`Unsupported dtype for WebGL fetch: ${t}`);return`
${n} ${o}(highp sampler2D tex, int idx) {
  ivec2 texSize = textureSize(tex, 0);
  int texel = idx / 4;
  int component = idx - texel * 4;
  ivec2 coord = ivec2(texel % texSize.x, texel / texSize.x);
  vec4 texVal = texelFetch(tex, coord, 0);
  float val;
  if (component == 0) val = texVal.x;
  else if (component == 1) val = texVal.y;
  else if (component == 2) val = texVal.z;
  else val = texVal.w;
  return ${e};
}
`}function W(t,o){switch(t){case E.Float32:return o;case E.Int32:return`intBitsToFloat(${o})`;case E.Uint32:return`uintBitsToFloat(${o})`;case E.Bool:return`intBitsToFloat(${o} ? 1 : 0)`;default:throw new Error(`Unsupported dtype for WebGL output: ${t}`)}}function I(t,o){switch(t){case E.Bool:return o?"true":"false";case E.Int32:return o.toString();case E.Uint32:return o.toString()+"u";case E.Float32:return Number.isNaN(o)?"uintBitsToFloat(0x7fc00000u)":Number.isFinite(o)?"float("+o.toString()+")":o>0?"uintBitsToFloat(0x7f800000u)":"uintBitsToFloat(0xff800000u)";default:throw new Error(`Unsupported dtype for WebGL constant: ${t}`)}}function b(t,o,n){const e=new Map,r=$=>{if(e.has($))return e.get($);const{op:a,src:u,dtype:x,arg:l}=$;let i="";if(B.Binary.has(a)){const s=r(u[0]),f=r(u[1]);if(a===c.Add)x===E.Bool?i=`(${s} || ${f})`:i=`(${s} + ${f})`;else if(a===c.Sub)i=`(${s} - ${f})`;else if(a===c.Mul)x===E.Bool?i=`(${s} && ${f})`:i=`(${s} * ${f})`;else if(a===c.Idiv)g(x)?i=`trunc(${s} / ${f})`:i=`(${s} / ${f})`;else if(a===c.Mod)g(x)?i=`(${s} - ${f} * trunc(${s} / ${f}))`:i=`(${s} % ${f})`;else if(a===c.Min)x===E.Bool?i=`(${s} && ${f})`:i=`min(${s}, ${f})`;else if(a===c.Max)x===E.Bool?i=`(${s} || ${f})`:i=`max(${s}, ${f})`;else if(a===c.BitCombine){let _=l==="and"?"&":l==="or"?"|":"^";x===E.Bool&&(_=_+_),i=`(${s} ${_} ${f})`}else a===c.BitShift&&(l==="shl"?i=`(${s} << ${f})`:i=`(${s} >> ${f})`)}else if(B.Compare.has(a)){const s=r(u[0]),f=r(u[1]);a===c.Cmplt?i=`(${s} < ${f})`:a===c.Cmpne&&(g(u[0].dtype)?i=`(${s} != ${f} || isnan(${s}) || isnan(${f}))`:i=`(${s} != ${f})`)}else if(B.Unary.has(a)){const s=r(u[0]);if(a===c.Sin)i=`sin(${d(s)})`;else if(a===c.Cos)i=`cos(${d(s)})`;else if(a===c.Asin)i=`asin(${d(s)})`;else if(a===c.Atan)i=`atan(${d(s)})`;else if(a===c.Exp)i=`exp(${d(s)})`;else if(a===c.Log)i=`log(${d(s)})`;else if(a===c.Erf)i=`erf(${d(s)})`;else if(a===c.Erfc)i=`erfc(${d(s)})`;else if(a===c.Sqrt)i=`sqrt(${d(s)})`;else if(a===c.Floor)i=`floor(${d(s)})`;else if(a===c.Ceil)i=`ceil(${d(s)})`;else if(a===c.Reciprocal)i=`(1.0 / ${s})`;else if(a===c.Cast)i=`${U(x)}(${d(s)})`;else if(a===c.Bitcast){const f=u[0].dtype;x===f?i=s:x===E.Float32?f===E.Int32?i=`intBitsToFloat(${d(s)})`:f===E.Uint32&&(i=`uintBitsToFloat(${d(s)})`):x===E.Int32?f===E.Float32?i=`floatBitsToInt(${d(s)})`:f===E.Uint32&&(i=`int(${d(s)})`):x===E.Uint32&&(f===E.Float32?i=`floatBitsToUint(${d(s)})`:f===E.Int32&&(i=`uint(${d(s)})`))}}else if(a===c.Threefry2x32){const[s,f,_,h]=u.map(A=>d(r(A))),m=l,T=`threefry2x32(uvec2(${s}, ${f}), uvec2(${_}, ${h}))`;m==="xor"?i=`(${T}.x ^ ${T}.y)`:m===0?i=`${T}.x`:m===1&&(i=`${T}.y`)}else if(a===c.Where){const[s,f,_]=u.map(r);i=`(${s} ? ${f} : ${_})`}else if(a===c.Const)i=I(x,l);else if(a===c.Special)i=l[0];else if(a===c.Variable)i=l;else if(a===c.GlobalIndex){const s=l[0],f=r(u[0]);i=`load_${n[s]}(${o[s]}, ${d(f)})`}if(!i)throw new O(a,x,"webgl",l);return e.set($,i),i};return r(t)}export{H as WebGLBackend};
