
import os
import glob
import numpy as np
import itk
import vtk
from vtk.util import numpy_support
import sys

# ============================================================================
# ПОИСК И ЗАГРУЗКА ДАННЫХ
# ============================================================================

def find_liver_datasets():

    print("\n" + "="*60)
    print(" ПОИСК ДАННЫХ СЕГМЕНТАЦИИ ПЕЧЕНИ")
    print("="*60)
    
    volume_dir = "volume"
    segmentation_dir = "segmentation"
    
    if not os.path.isdir(volume_dir):
        print(f" Папка '{volume_dir}' не найдена")
        return []
    
    if not os.path.isdir(segmentation_dir):
        print(f" Папка '{segmentation_dir}' не найдена")
        return []
    
    # Ищем файлы volume
    volume_files = glob.glob(os.path.join(volume_dir, "volume-*.nii"))
    segmentation_files = glob.glob(os.path.join(segmentation_dir, "segmentation-*.nii"))
    
    if not volume_files:
        print(f" Нет файлов volume-*.nii в папке '{volume_dir}'")
        return []
    
    print(f" Найдено volume файлов: {len(volume_files)}")
    print(f" Найдено segmentation файлов: {len(segmentation_files)}")
    
    # Создаем пары (volume, segmentation)
    datasets = []
    
    for volume_file in volume_files:

        basename = os.path.basename(volume_file)
        number = basename.replace("volume-", "").replace(".nii", "")
        
        seg_file = os.path.join(segmentation_dir, f"segmentation-{number}.nii")
        
        if os.path.isfile(seg_file):
            datasets.append({
                'number': number,
                'volume': volume_file,
                'segmentation': seg_file,
                'name': f"Печень #{number}"
            })
        else:
            print(f"Для volume-{number}.nii не найдена segmentation-{number}.nii")

    datasets.sort(key=lambda x: int(x['number']))
    
    print(f"\nНайдено полных пар: {len(datasets)}")
    for i, ds in enumerate(datasets[:5], 1):
        print(f" {i}. {ds['name']}")
    
    if len(datasets) > 5:
        print(f" ... и еще {len(datasets)-5} пар")
    
    return datasets

def load_liver_data(volume_file, segmentation_file):

    number = os.path.basename(volume_file).replace("volume-", "").replace(".nii", "")
    print(f"\n Загружаем сканирование: Печень #{number}")
    
    try:
        # Загружаем volume
        volume_image = itk.imread(volume_file)
        volume_array = itk.array_view_from_image(volume_image)
        size = itk.size(volume_image)
        spacing = volume_image.GetSpacing()
        
        print(f" ✓ VOLUME | Размер: {size} | Spacing: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f} мм")
        
        # Загружаем segmentation
        seg_image = None
        seg_array = None
        
        if os.path.isfile(segmentation_file):
            try:
                seg_image = itk.imread(segmentation_file)
                seg_array = itk.array_view_from_image(seg_image)
                seg_size = itk.size(seg_image)
                print(f" ✓ SEGMENTATION | Размер: {seg_size}")
                
                # Проверяем размеры
                if size != seg_size:
                    print(f" Размеры не совпадают: volume {size} vs segmentation {seg_size}")
            except Exception as e:
                print(f"  SEGMENTATION | Ошибка загрузки: {e}")
                seg_image = None
        else:
            print(f"  Файл segmentation не найден")
        
        data = {
            'number': number,
            'volume': volume_image,
            'volume_array': volume_array,
            'segmentation': seg_image,
            'segmentation_array': seg_array,
            'spacing': spacing,
            'size': size
        }
        
        print(f"Данные загружены успешно")
        return data
        
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None

# ============================================================================
# 2D ВИЗУАЛИЗАЦИЯ
# ============================================================================

def visualize_interactive_slices(data, dataset_name=""):

    print(f"\n" + "="*60)
    print(f" 2D ВИЗУАЛИЗАЦИЯ: {dataset_name}")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
    except ImportError:
        print(" Требуется matplotlib: pip install matplotlib")
        return
    
    volume_image = data['volume']
    volume_array = data['volume_array']
    seg_array = data.get('segmentation_array')
    spacing = data['spacing']
    
    # Определяем количество срезов (глубина по Z)
    if volume_array.ndim == 4:
        num_slices = volume_array.shape[3]
    else:
        num_slices = volume_array.shape[2]
    
    # Ищем срезы с печенью
    liver_slices = []
    if seg_array is not None:
        for z in range(num_slices):
            if seg_array.ndim == 4:
                if np.any(seg_array[0, :, :, z] > 0):
                    liver_slices.append(z)
            else:
                if np.any(seg_array[:, :, z] > 0):
                    liver_slices.append(z)
    
    print(f" Всего срезов: {num_slices}")
    if liver_slices:
        print(f"Срезы с печенью: {len(liver_slices)} (диапазон: {min(liver_slices)}-{max(liver_slices)})")
        initial_slice = liver_slices[0]
    else:
        print(" Печень не найдена в маске")
        initial_slice = num_slices // 2
    
    print(f" Физические размеры вокселя: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} мм")
    
    # Создаем окно для визуализации
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(bottom=0.25, top=0.9)
    
    def prepare_slice(image, idx):

        array = itk.array_view_from_image(image)
        dim = image.GetImageDimension()
        
        if dim == 4:
            slice_data = array[0, :, :, idx]
        else:
            slice_data = array[:, :, idx]
        
        # Транспонируем и переворачиваем для правильной ориентации
        slice_data = slice_data.T
        slice_data = np.flipud(slice_data)
        
        return slice_data
    
    def update_slice(slice_idx):

        ax1.clear()
        ax2.clear()
        
        # Левая панель: CT изображение
        ct_slice = prepare_slice(volume_image, slice_idx)
        
        # Нормализация CT
        if ct_slice.max() > ct_slice.min():
            ct_slice_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
        else:
            ct_slice_norm = ct_slice
        
        ax1.imshow(ct_slice_norm, cmap='gray', aspect='auto')
        ax1.set_title(f'CT - Срез {slice_idx}')
        ax1.axis('off')
        
        # Правая панель: CT + сегментация печени
        ax2.imshow(ct_slice_norm, cmap='gray', aspect='auto')
        
        # Накладываем сегментацию 
        if seg_array is not None:
            seg_slice = prepare_slice(data['segmentation'], slice_idx)
            
            # Создаем цветовую маску для печени
            mask_rgb = np.zeros((*seg_slice.shape, 4))
            
            # Основной цвет печени
            liver_mask = (seg_slice > 0)
            
            if liver_mask.any():
                # Зеленый цвет для печени
                mask_rgb[liver_mask, 0] = 0.0
                mask_rgb[liver_mask, 1] = 1.0
                mask_rgb[liver_mask, 2] = 0.0
                mask_rgb[liver_mask, 3] = 0.6
                
                ax2.imshow(mask_rgb, aspect='auto')
                ax2.set_title('CT + Сегментация ✅ ПЕЧЕНЬ ВИДНА', color='green', fontweight='bold')
            else:
                ax2.set_title('CT + Сегментация ⚠ НЕТ ПЕЧЕНИ', color='black')
        else:
            ax2.set_title('CT (нет сегментации)')
        
        ax2.axis('off')
        
        # Подсвечиваем рамку если на срезе есть печень
        if slice_idx in liver_slices:
            for spine in ax1.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
            for spine in ax2.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        
        plt.draw()
    
    # Инициализируем первый срез
    update_slice(initial_slice)
    
    # Слайдер
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Срез', 0, num_slices-1, valinit=initial_slice, valstep=1)
    slider.on_changed(lambda val: update_slice(int(val)))
    
    # Кнопки
    if liver_slices:
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.8, 0.05, 0.1, 0.04])
        ax_reset = plt.axes([0.4, 0.05, 0.2, 0.04])
        
        def go_prev(event):
            current = int(slider.val)
            prev = [s for s in liver_slices if s < current]
            if prev:
                slider.set_val(prev[-1])
        
        def go_next(event):
            current = int(slider.val)
            nxt = [s for s in liver_slices if s > current]
            if nxt:
                slider.set_val(nxt[0])
        
        def go_first_liver(event):
            if liver_slices:
                slider.set_val(liver_slices[0])
        
        btn_prev = Button(ax_prev, '← Пред.')
        btn_next = Button(ax_next, 'След. →')
        btn_reset = Button(ax_reset, 'К первому срезу')
        
        btn_prev.on_clicked(go_prev)
        btn_next.on_clicked(go_next)
        btn_reset.on_clicked(go_first_liver)
    
    plt.suptitle(f'Печень {dataset_name} | Срезов: {num_slices} | Spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} мм')
    
    # Сохраняем скриншот
    output_file = f"2d_slices_liver_{dataset_name.replace('/', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Сохранено: {output_file}")
    
    plt.show()

# ============================================================================
# 3D ВИЗУАЛИЗАЦИЯ ПЕЧЕНИ 
# ============================================================================

def visualize_3d_liver_simple(data, dataset_name=""):

    print(f"\n" + "="*60)
    print(f" 3D ВИЗУАЛИЗАЦИЯ ПЕЧЕНИ: {dataset_name}")
    print("="*60)
    
    if data.get('segmentation_array') is None:
        print(" Нет маски для 3D визуализации")
        return
    
    seg_array = data['segmentation_array']
    spacing = data['spacing']
    
    if np.max(seg_array) == 0:
        print(" Маска пустая - нет печени для визуализации")
        return
    
    liver_voxels = np.sum(seg_array > 0)
    print(f" Вокселей печени: {liver_voxels:,}")
    print(f" Оригинальный spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} мм")
    
    try:

        # Берем наибольший spacing как эталон
        max_spacing = max(spacing)
        normalized_spacing = tuple(s / max_spacing for s in spacing)
        
        print(f" Нормализованный spacing: {normalized_spacing[0]:.3f}×{normalized_spacing[1]:.3f}×{normalized_spacing[2]:.3f}")
        
        # Конвертируем numpy массив в VTK ImageData
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(seg_array.shape)
        
        # Используем нормализованный spacing вместо оригинального
        vtk_data.SetSpacing(normalized_spacing[0], normalized_spacing[1], normalized_spacing[2])
        
        flat_data = seg_array.flatten(order='F')
        vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_SHORT)
        vtk_data.GetPointData().SetScalars(vtk_array)
        
        # Пороговая обработка для извлечения печени
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputData(vtk_data)
        threshold.ThresholdByLower(0.5)
        threshold.SetInValue(255)
        threshold.SetOutValue(0)
        threshold.Update()
        
        # Извлечение поверхности (Marching Cubes)
        contour = vtk.vtkMarchingCubes()
        contour.SetInputConnection(threshold.GetOutputPort())
        contour.ComputeNormalsOn()
        contour.SetValue(0, 128)
        contour.Update()
        
        # Сглаживание поверхности
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(30)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()
        
        # Маппер использует сглаженные данные
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        # Актер
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.6, 0.8, 0.4)
        actor.GetProperty().SetOpacity(0.95)
        actor.GetProperty().SetDiffuse(0.9)
        actor.GetProperty().SetSpecular(0.6)
        actor.GetProperty().SetSpecularPower(10)
        
        # Создаем рендерер
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.2)
        renderer.AddActor(actor)
        
        # Адаптивная камера на основе размеров объекта
        renderer.ResetCamera()
        bounds = actor.GetBounds()
        
        diagonal = np.sqrt((bounds[1]-bounds[0])**2 + 
                          (bounds[3]-bounds[2])**2 + 
                          (bounds[5]-bounds[4])**2)
        
        camera = renderer.GetActiveCamera()
        camera.SetDistance(diagonal * 1.5)
        camera.Azimuth(30)
        camera.Elevation(20)
        
        # Создаем окно рендеринга
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(900, 700)
        render_window.SetWindowName(f"3D Печень: {dataset_name}")
        
        # Интерактор
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        # Добавляем инструкции
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput("ЛКМ: вращение | Колесико: масштаб | ПКМ: панорама")
        text_prop = text_actor.GetTextProperty()
        text_prop.SetFontSize(12)
        text_prop.SetColor(1, 1, 1)
        text_actor.SetPosition(10, 10)
        renderer.AddActor2D(text_actor)
        
        print(f" Размеры печени: {diagonal:.1f} мм (нормализованные)")
        print("\n Управление:")
        print(" • Вращение: левая кнопка мыши + движение")
        print(" • Приближение/отдаление: колесико мыши")
        print(" • Панорамирование: правая кнопка мыши + движение")
        print(" • Выход: нажмите 'q' или закройте окно")
        
        # Запускаем
        interactor.Initialize()
        render_window.Render()
        interactor.Start()
        
    except Exception as e:
        print(f" Ошибка 3D визуализации: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 3D ВИЗУАЛИЗАЦИЯ С ДЕТАЛЯМИ (ИСПРАВЛЕННАЯ)
# ============================================================================

def visualize_3d_liver_detailed(data, dataset_name=""):

    print(f"\n" + "="*60)
    print(f" 3D ДЕТАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ПЕЧЕНИ: {dataset_name}")
    print("="*60)
    
    if data.get('segmentation_array') is None or data.get('volume_array') is None:
        print(" Нет необходимых данных для 3D визуализации")
        return
    
    seg_array = data['segmentation_array']
    volume_array = data['volume_array']
    spacing = data['spacing']
    
    if np.max(seg_array) == 0:
        print(" Маска пустая")
        return
    
    print(f" Вокселей печени: {np.sum(seg_array > 0):,}")
    print(f" Оригинальный spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} мм")
    
    try:

        max_spacing = max(spacing)
        normalized_spacing = tuple(s / max_spacing for s in spacing)
        
        print(f" Нормализованный spacing: {normalized_spacing[0]:.3f}×{normalized_spacing[1]:.3f}×{normalized_spacing[2]:.3f}")
        
        # Подготовка VTK данных с нормализованным масштабом
        vtk_seg = vtk.vtkImageData()
        vtk_seg.SetDimensions(seg_array.shape)
        vtk_seg.SetSpacing(normalized_spacing[0], normalized_spacing[1], normalized_spacing[2])
        
        flat_seg = seg_array.flatten(order='F')
        vtk_seg_array = numpy_support.numpy_to_vtk(flat_seg, deep=True, array_type=vtk.VTK_SHORT)
        vtk_seg.GetPointData().SetScalars(vtk_seg_array)
        
        # Volume для контекста
        vtk_volume = vtk.vtkImageData()
        vtk_volume.SetDimensions(volume_array.shape)
        vtk_volume.SetSpacing(normalized_spacing[0], normalized_spacing[1], normalized_spacing[2])
        
        flat_volume = volume_array.flatten(order='F')
        vtk_volume_array = numpy_support.numpy_to_vtk(flat_volume, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_volume.GetPointData().SetScalars(vtk_volume_array)
        
        # Создаем 3 рендерера
        renderers = []
        
        view_configs = [
            ("Печень + контекст", True, 0.7),
            ("Только печень", False, 0.95),
            ("Печень прозрачная", False, 0.4)
        ]
        
        for title, show_context, opacity in view_configs:
            renderer = vtk.vtkRenderer()
            renderer.SetBackground(0.05, 0.05, 0.1)
            
            # Если нужен контекст - добавляем окружающие ткани
            if show_context:
                threshold_context = vtk.vtkImageThreshold()
                threshold_context.SetInputData(vtk_volume)
                threshold_context.ThresholdBetween(200, volume_array.max())
                threshold_context.SetInValue(255)
                threshold_context.SetOutValue(0)
                threshold_context.Update()
                
                contour_context = vtk.vtkMarchingCubes()
                contour_context.SetInputConnection(threshold_context.GetOutputPort())
                contour_context.ComputeNormalsOn()
                contour_context.SetValue(0, 128)
                contour_context.Update()
                
                # Сглаживание для контекста
                smoother_context = vtk.vtkSmoothPolyDataFilter()
                smoother_context.SetInputConnection(contour_context.GetOutputPort())
                smoother_context.SetNumberOfIterations(20)
                smoother_context.SetRelaxationFactor(0.08)
                smoother_context.Update()
                
                mapper_context = vtk.vtkPolyDataMapper()
                mapper_context.SetInputConnection(smoother_context.GetOutputPort())
                mapper_context.ScalarVisibilityOff()
                
                actor_context = vtk.vtkActor()
                actor_context.SetMapper(mapper_context)
                actor_context.GetProperty().SetColor(0.7, 0.7, 0.7)
                actor_context.GetProperty().SetOpacity(0.15)
                actor_context.GetProperty().SetDiffuse(0.8)
                
                renderer.AddActor(actor_context)
            
            # Добавляем печень
            threshold = vtk.vtkImageThreshold()
            threshold.SetInputData(vtk_seg)
            threshold.ThresholdByLower(0.5)
            threshold.SetInValue(255)
            threshold.SetOutValue(0)
            threshold.Update()
            
            contour = vtk.vtkMarchingCubes()
            contour.SetInputConnection(threshold.GetOutputPort())
            contour.ComputeNormalsOn()
            contour.SetValue(0, 128)
            contour.Update()
            
            # Сглаживание для печени
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(contour.GetOutputPort())
            smoother.SetNumberOfIterations(30)
            smoother.SetRelaxationFactor(0.1)
            smoother.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(smoother.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.6, 0.8, 0.4)
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().SetDiffuse(0.8)
            actor.GetProperty().SetSpecular(0.5)
            
            renderer.AddActor(actor)
            
            # Адаптивная камера для каждого вида
            renderer.ResetCamera()
            bounds = actor.GetBounds()
            diagonal = np.sqrt((bounds[1]-bounds[0])**2 + 
                              (bounds[3]-bounds[2])**2 + 
                              (bounds[5]-bounds[4])**2)
            
            camera = renderer.GetActiveCamera()
            camera.SetDistance(diagonal * 1.5)
            
            if title == "Печень + контекст":
                camera.Azimuth(30)
                camera.Elevation(20)
            elif title == "Только печень":
                camera.Azimuth(45)
                camera.Elevation(30)
            else:  # Печень прозрачная
                camera.Azimuth(60)
                camera.Elevation(15)
            
            renderers.append(renderer)
        
        # Создаем окно с 3 вьюпортами
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(1400, 480)
        render_window.SetWindowName(f"3D Детальная визуализация печени: {dataset_name}")
        
        for i, renderer in enumerate(renderers):
            render_window.AddRenderer(renderer)
            renderer.SetViewport(i/3, 0, (i+1)/3, 1)
        
        # Интерактор
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        print("\n Описание окон (слева направо):")
        for i, (title, _, _) in enumerate(view_configs):
            print(f" {i+1}. {title}")
        
        print("\n Управление (в каждом окне отдельно):")
        print(" • Вращение: левая кнопка мыши")
        print(" • Приближение: колесико мыши")
        print(" • Панорамирование: правая кнопка мыши")
        
        # Запускаем
        interactor.Initialize()
        render_window.Render()
        interactor.Start()
        
    except Exception as e:
        print(f" Ошибка 3D визуализации: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# СТАТИСТИКА
# ============================================================================

def calculate_liver_statistics(data, dataset_name=""):
    """
    Сатистикапечени:
    - Объемы печени
    - Размеры в мл и см³
    - Размеры в мм и см
    - Интенсивность (HU для CT)
    """
    print(f"\n" + "="*60)
    print(f" СТАТИСТИКА ПЕЧЕНИ: {dataset_name}")
    print("="*60)
    
    if data.get('segmentation_array') is None:
        print(" Нет маски для расчета статистики")
        return
    
    seg_array = data['segmentation_array']
    volume_array = data.get('volume_array')
    spacing = data['spacing']
    
    # Объем вокселя (используем оригинальный spacing для корректного объема)
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # Общая статистика печени
    liver_voxels = np.sum(seg_array > 0)
    
    if liver_voxels == 0:
        print(" Печень не найдена в маске")
        return
    
    liver_volume_mm3 = liver_voxels * voxel_volume_mm3
    liver_volume_ml = liver_volume_mm3 / 1000.0
    liver_volume_cm3 = liver_volume_mm3 / 1000.0
    
    # Координаты границ печени
    liver_coords = np.argwhere(seg_array > 0)
    min_coords = liver_coords.min(axis=0)
    max_coords = liver_coords.max(axis=0)
    
    # Размеры в мм
    size_mm = (max_coords - min_coords + 1) * spacing
    size_cm = size_mm / 10.0
    
    print("\n Размеры вокселя и изображения:")
    print(f" • Разрешение: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} мм")
    print(f" • Объем вокселя: {voxel_volume_mm3:.3f} мм³")
    print(f" • Размеры изображения: {seg_array.shape}")
    
    print("\n ОБЪЕМ ПЕЧЕНИ:")
    print("-" * 50)
    print(f" • Вокселей: {liver_voxels:,}")
    print(f" • Объем: {liver_volume_ml:.2f} мл ({liver_volume_cm3:.2f} см³)")
    print(f" • В литрах: {liver_volume_ml/1000:.4f} л")
    
    print("\n РАЗМЕРЫ ПЕЧЕНИ:")
    print("-" * 50)
    print(f" • В мм: {size_mm[0]:.1f} × {size_mm[1]:.1f} × {size_mm[2]:.1f} мм")
    print(f" • В см: {size_cm[0]:.1f} × {size_cm[1]:.1f} × {size_cm[2]:.1f} см")
    
    # Статистика интенсивности печени
    if volume_array is not None:
        liver_intensities = volume_array[seg_array > 0]
        
        print("\n ИНТЕНСИВНОСТЬ (СТАТИСТИКА):")
        print("-" * 50)
        print(f" • Минимум: {liver_intensities.min():.1f}")
        print(f" • Максимум: {liver_intensities.max():.1f}")
        print(f" • Среднее: {liver_intensities.mean():.1f}")
        print(f" • Медиана: {np.median(liver_intensities):.1f}")
        print(f" • Стандартное отклонение: {liver_intensities.std():.1f}")
    
    # Сохраняем в файл
    stats_file = f"liver_statistics_{dataset_name.replace('/', '_')}.txt"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"СТАТИСТИКА ПЕЧЕНИ: {dataset_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("ПАРАМЕТРЫ ВОКСЕЛЯ:\n")
        f.write(f"Разрешение: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} мм\n")
        f.write(f"Объем вокселя: {voxel_volume_mm3:.3f} мм³\n\n")
        
        f.write("ОБЪЕМ ПЕЧЕНИ:\n")
        f.write(f"Вокселей: {liver_voxels:,}\n")
        f.write(f"Объем: {liver_volume_ml:.2f} мл\n")
        f.write(f"Объем: {liver_volume_cm3:.2f} см³\n")
        f.write(f"Объем: {liver_volume_ml/1000:.4f} л\n\n")
        
        f.write("РАЗМЕРЫ ПЕЧЕНИ:\n")
        f.write(f"В мм: {size_mm[0]:.1f} × {size_mm[1]:.1f} × {size_mm[2]:.1f} мм\n")
        f.write(f"В см: {size_cm[0]:.1f} × {size_cm[1]:.1f} × {size_cm[2]:.1f} см\n")
        
        if volume_array is not None:
            f.write("\nСТАТИСТИКА ИНТЕНСИВНОСТИ:\n")
            f.write(f"Минимум: {liver_intensities.min():.1f}\n")
            f.write(f"Максимум: {liver_intensities.max():.1f}\n")
            f.write(f"Среднее: {liver_intensities.mean():.1f}\n")
            f.write(f"Стандартное отклонение: {liver_intensities.std():.1f}\n")
    
    print(f"\n Статистика сохранена: {stats_file}")

# ============================================================================
# УПРАВЛЕНИЕ ПРОГРАММОЙ
# ============================================================================

def main():

    print("\n" + "="*70)
    print(" LIVER SEGMENTATION VISUALIZER - ФИНАЛЬНАЯ ВЕРСИЯ")
    print("="*70)
    
    print("\nПоддерживаемая структура данных:")
    print(" • volume/ - папка с файлами volume-*.nii")
    print(" • segmentation/ - папка с файлами segmentation-*.nii")
    
    print("\nФИНАЛЬНЫЕ УЛУЧШЕНИЯ:")
    print("  Нормализация масштаба вокселей (избегает растяжения)")
    print("  Правильный aspect ratio в 2D визуализации")
    print("  Сглаживание поверхности печени")
    print("  Адаптивная камера на основе размеров объекта")
    print("  Улучшенная контрастность маски")
    print("  Правильная 3D геометрия без искажений")
    
    print("\nТребуемые библиотеки:")
    print(" • itk, vtk, numpy, matplotlib")
    print("="*70)
    
    # Ищем наборы данных
    datasets = find_liver_datasets()
    
    if not datasets:
        print("\n Данные о печени не найдены!")
        print("\nПоместите данные в текущую директорию:")
        print(" /ваша_папка/")
        print(" ├── volume/")
        print(" │   ├── volume-0.nii")
        print(" │   ├── volume-1.nii")
        print(" │   └── ...")
        print(" └── segmentation/")
        print("     ├── segmentation-0.nii")
        print("     ├── segmentation-1.nii")
        print("     └── ...")
        return
    
    # Загружаем первый набор данных
    current_idx = 0
    current_dataset = datasets[current_idx]
    data = load_liver_data(current_dataset['volume'], current_dataset['segmentation'])
    
    if not data:
        print(" Не удалось загрузить первый набор данных")
        return
    
    # Основной цикл программы
    while True:
        print("\n" + "="*70)
        print(" ГЛАВНОЕ МЕНЮ")
        print("="*70)
        print(f" Текущий набор: {current_dataset['name']} ({current_idx+1}/{len(datasets)})")
        print("-"*70)
        print("1.  Информация о печени")
        print("2.  2D визуализация (интерактивные срезы)")
        print("3.  3D визуализация печени")
        print("4.  3D детальная визуализация")
        print("5.  Статистика печени")
        print("6.  Всё сразу (2D + 3D + статистика)")
        print("7.  Следующий набор данных")
        print("8.  Предыдущий набор данных")
        print("9.  Выход")
        print("="*70)
        
        try:
            choice = input("\nВыберите действие (1-9): ").strip()
            
            if choice == '1':
                # Информация о печени
                print(f"\nИНФОРМАЦИЯ О ПЕЧЕНИ:")
                print(f" Номер: {current_dataset['number']}")
                print(f" Путь volume: {current_dataset['volume']}")
                print(f" Путь segmentation: {current_dataset['segmentation']}")
                print(f" Размер изображения: {data['size']}")
                print(f" Spacing: {data['spacing']}")
                
                if data.get('segmentation_array') is not None:
                    seg_array = data['segmentation_array']
                    liver_voxels = np.sum(seg_array > 0)
                    print(f" Вокселей печени: {liver_voxels:,}")
            
            elif choice == '2':
                # 2D визуализация
                visualize_interactive_slices(data, current_dataset['name'])
            
            elif choice == '3':
                # 3D печени простая
                visualize_3d_liver_simple(data, current_dataset['name'])
            
            elif choice == '4':
                # 3D печени детальная
                visualize_3d_liver_detailed(data, current_dataset['name'])
            
            elif choice == '5':
                # Статистика
                calculate_liver_statistics(data, current_dataset['name'])
            
            elif choice == '6':
                # Всё сразу
                print(f"\n ЗАПУСК ВСЕХ ВИЗУАЛИЗАЦИЙ ДЛЯ {current_dataset['name']}...")
                visualize_interactive_slices(data, current_dataset['name'])
                visualize_3d_liver_simple(data, current_dataset['name'])
                visualize_3d_liver_detailed(data, current_dataset['name'])
                calculate_liver_statistics(data, current_dataset['name'])
                print(f"\n Все визуализации завершены!")
            
            elif choice == '7':
                # Следующий набор
                if len(datasets) > 1:
                    current_idx = (current_idx + 1) % len(datasets)
                    current_dataset = datasets[current_idx]
                    print(f"\n Загружаем: {current_dataset['name']}")
                    data = load_liver_data(current_dataset['volume'], current_dataset['segmentation'])
                    if not data:
                        print(" Ошибка загрузки, возвращаемся к предыдущему")
                        current_idx = (current_idx - 1) % len(datasets)
                        current_dataset = datasets[current_idx]
                        data = load_liver_data(current_dataset['volume'], current_dataset['segmentation'])
                else:
                    print(" Больше наборов данных нет")
            
            elif choice == '8':
                # Предыдущий набор
                if len(datasets) > 1:
                    current_idx = (current_idx - 1) % len(datasets)
                    current_dataset = datasets[current_idx]
                    print(f"\n Загружаем: {current_dataset['name']}")
                    data = load_liver_data(current_dataset['volume'], current_dataset['segmentation'])
                    if not data:
                        print(" Ошибка загрузки, возвращаемся к следующему")
                        current_idx = (current_idx + 1) % len(datasets)
                        current_dataset = datasets[current_idx]
                        data = load_liver_data(current_dataset['volume'], current_dataset['segmentation'])
                else:
                    print(" Других наборов данных нет")
            
            elif choice == '9':
                # Выход
                print("\n Выход из программы")
                break
            
            else:
                print(" Неверный выбор. Введите число от 1 до 9.")
        
        except KeyboardInterrupt:
            print("\n\n Программа прервана")
            break
        except Exception as e:
            print(f" Ошибка: {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# ЗАПУСК ПРОГРАММЫ
# ============================================================================

if __name__ == "__main__":
    main()
