import fs from 'fs'; import path from 'path'; import { fileURLToPath } from 'url';
import { buildGrid } from '../src/grid.js';
import { buildMaterialGrid, computeMaterialComponents, applyPECPlates } from '../src/materials.js';
import { computeGeneralCoefficients, applyLumpedElementCoefficients } from '../src/coefficients.js';
import { initCPML } from '../src/cpml.js';
import { initWaveforms, initVoltageSources } from '../src/sources.js';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dx=0.262e-3,dy=0.4e-3,dz=0.4e-3,courantFactor=0.9,numberOfTimeSteps=200,numberOfCellsPerWavelength=20;
const boundary={type_xn:'cpml',type_xp:'cpml',type_yn:'cpml',type_yp:'cpml',type_zn:'cpml',type_zp:'cpml',air_buffer_xn:10,air_buffer_xp:10,air_buffer_yn:10,air_buffer_yp:10,air_buffer_zn:10,air_buffer_zp:10,cpml_cells_xn:8,cpml_cells_xp:8,cpml_cells_yn:8,cpml_cells_yp:8,cpml_cells_zn:8,cpml_cells_zp:8,cpml_order:3,cpml_sigma_factor:1.3,cpml_kappa_max:7,cpml_alpha_min:0,cpml_alpha_max:0.05};
const materialTypes=[{eps_r:1,mu_r:1,sigma_e:0,sigma_m:0},{eps_r:1,mu_r:1,sigma_e:1e10,sigma_m:0},{eps_r:1,mu_r:1,sigma_e:0,sigma_m:1e10},{eps_r:2.2,mu_r:1,sigma_e:0,sigma_m:0}];
const bricks=[{min_x:-0.787e-3,min_y:0,min_z:0,max_x:0,max_y:40e-3,max_z:40e-3,material_type:4},{min_x:0,min_y:0,min_z:24e-3,max_x:0,max_y:28.4e-3,max_z:26.4e-3,material_type:2},{min_x:0,min_y:16e-3,min_z:30e-3,max_x:0,max_y:28.4e-3,max_z:32.4e-3,material_type:2},{min_x:0,min_y:26e-3,min_z:8.4e-3,max_x:0,max_y:28.4e-3,max_z:32.4e-3,material_type:2},{min_x:0,min_y:20.8e-3,min_z:16e-3,max_x:0,max_y:23.2e-3,max_z:32.4e-3,material_type:2},{min_x:-0.787e-3,min_y:16e-3,min_z:30e-3,max_x:0,max_y:16e-3,max_z:32.4e-3,material_type:2},{min_x:-0.787e-3,min_y:0,min_z:0,max_x:-0.787e-3,max_y:16e-3,max_z:40e-3,material_type:2}];
const grid=buildGrid({dx,dy,dz,boundary,bricks,spheres:[],courantFactor,numberOfTimeSteps});
const matGrid=buildMaterialGrid(grid,bricks,[]);
const matComps=computeMaterialComponents(matGrid,materialTypes,grid);
applyPECPlates(bricks,materialTypes,matComps,grid);
let coeffs=computeGeneralCoefficients(matComps,grid);
function scan(label,obj){let total=0;for(const k of Object.keys(obj)){const a=obj[k];if(!a||!a.length)continue;let c=0,first=-1;for(let n=0;n<a.length;n++){if(!Number.isFinite(a[n])){c++;if(first<0)first=n;}}if(c){console.log(`  ${label}.${k}: ${c} non-finite, first @${first} = ${a[first]}`);total+=c;}}return total;}
console.log('== after computeGeneralCoefficients ==');
let t=scan('coeffs',coeffs);
console.log(`(matComps scan)`); scan('matComps',matComps);
const cpml=initCPML(boundary,coeffs,grid);
console.log('== after initCPML (coeffs divided by kappa) ==');
scan('coeffs',coeffs);
console.log('== CPML face arrays ==');
for(const face of ['xn','xp','yn','yp','zn','zp']){if(cpml[face])scan('cpml.'+face,cpml[face]);}
let waveforms={gaussian:[{number_of_cells_per_wavelength:0},{number_of_cells_per_wavelength:15}]};
waveforms=initWaveforms(waveforms,numberOfTimeSteps,grid.dt,numberOfCellsPerWavelength,[dx,dy,dz]);
const voltageSources=[{min_x:-0.787e-3,min_y:0,min_z:24e-3,max_x:0,max_y:0,max_z:26.4e-3,direction:'xp',resistance:50,magnitude:1,waveform_type:'gaussian',waveform_index:1}];
initVoltageSources(voltageSources,waveforms,grid);
applyLumpedElementCoefficients(coeffs,matComps,grid,voltageSources,[],[],[],[],[]);
console.log('== after applyLumpedElementCoefficients ==');
scan('coeffs',coeffs);
// Decode the offending Hx index from earlier
const i=Math.floor(734528/(grid.ny*grid.nz)), r=734528%(grid.ny*grid.nz), j=Math.floor(r/grid.nz), k=r%grid.nz;
console.log(`Hx[734528] -> i=${i} j=${j} k=${k}  (nx=${grid.nx} ny=${grid.ny} nz=${grid.nz}, nzp1=${grid.nzp1})`);
console.log('Chxh =',coeffs.Chxh[734528],' Chxey =',coeffs.Chxey[734528],' Chxez =',coeffs.Chxez[734528]);
