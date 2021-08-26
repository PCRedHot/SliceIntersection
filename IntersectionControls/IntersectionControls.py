import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np


#
# IntersectionControls
#

class IntersectionControls(ScriptedLoadableModule):
	"""Uses ScriptedLoadableModule base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self, parent):
		ScriptedLoadableModule.__init__(self, parent)
		self.parent.title = "IntersectionControls"
		self.parent.categories = [
			"Slice"]
		self.parent.dependencies = []
		self.parent.contributors = [
			"Parry Choi (HKU)"]
		self.parent.helpText = ""
		self.parent.acknowledgementText = ""

		# Additional initialization step after application startup is complete
		slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
	"""
	Add data sets to Sample Data module.
	"""
	# It is always recommended to provide sample data for users to make it easy to try the module,
	# but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

	import SampleData
	iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

	# To ensure that the source code repository remains small (can be downloaded and installed quickly)
	# it is recommended to store data sets that are larger than a few MB in a Github release.

	# IntersectionControls1
	SampleData.SampleDataLogic.registerCustomSampleDataSource(
		# Category and sample name displayed in Sample Data module
		category='IntersectionControls',
		sampleName='IntersectionControls1',
		# Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
		# It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
		thumbnailFileName=os.path.join(iconsPath, 'IntersectionControls1.png'),
		# Download URL and target file name
		uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
		fileNames='IntersectionControls1.nrrd',
		# Checksum to ensure file integrity. Can be computed by this command:
		#  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
		checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
		# This node name will be used when the data set is loaded
		nodeNames='IntersectionControls1'
	)

	# IntersectionControls2
	SampleData.SampleDataLogic.registerCustomSampleDataSource(
		# Category and sample name displayed in Sample Data module
		category='IntersectionControls',
		sampleName='IntersectionControls2',
		thumbnailFileName=os.path.join(iconsPath, 'IntersectionControls2.png'),
		# Download URL and target file name
		uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
		fileNames='IntersectionControls2.nrrd',
		checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
		# This node name will be used when the data set is loaded
		nodeNames='IntersectionControls2'
	)


#
# IntersectionControlsWidget
#

class IntersectionControlsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
	"""Uses ScriptedLoadableModuleWidget base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""
	
	class IntersectionVariables:
		def __init__(self):
			self.coreKeyPressing = False  # True if pressing original key from application, to disable any processing
			self.leftMousePressing = False
			self.colourSelected = None  # not None if translation mode
			self.rotationMode = False  # True for rotation mode, False for translation mode
			self.rotationCenter = None  # Center of Rotation on XY plane (intersection point)
			self.rotationCenterRAS = None  # Center of Rotation in RAS coordinate
			self.rotationPrevAngleRad = None  # previous angle of mouse relative to intersection point in XY coordinate in Radian
			self.endpoints = None
		
		def processInit(self, caller=None, colour=None):
			self.coreKeyPressing = False
			self.leftMousePressing = False
			self.colourSelected = None
			self.rotationMode = False
			self.rotationCenter = None
			self.rotationCenterRAS = None
			self.rotationPrevAngleRad = None
			self.endpoints = None
		
		def updateEndpoints(self, caller=None, colour=None):
			if colour is None or caller is None:
				self.endpoints = None
			if colour is not None and caller is not None:
				sliceNode = caller.logic.sliceColourToNode[colour]
				m_XY_to_RAS = caller.getNumpyMatrixFromVTK4x4(sliceNode.GetXYToRAS())
				m_RAS_to_XY = np.linalg.inv(m_XY_to_RAS)
				
				slicePlaneNormal = [0, 0, 1]
				slicePlaneOrigin = [0, 0, 0]
				
				self.endpoints = []
				for c in caller.logic.sliceColourToID.keys():
					if c == colour:
						continue
					intersectingSliceNode = caller.logic.sliceColourToNode[c]
					intersectingSliceDimension = intersectingSliceNode.GetDimensions()
					intersecting_m_XY_to_RAS = caller.getNumpyMatrixFromVTK4x4(intersectingSliceNode.GetXYToRAS())
					
					m_IntersectingXY_to_XY = np.matmul(m_RAS_to_XY, intersecting_m_XY_to_RAS)
					
					intersectingPlaneOrigin = np.matmul(m_IntersectingXY_to_XY, [0, 0, 0, 1])
					intersectingPlaneX = np.matmul(m_IntersectingXY_to_XY, [intersectingSliceDimension[0], 0, 0, 1])
					intersectingPlaneY = np.matmul(m_IntersectingXY_to_XY, [0, intersectingSliceDimension[1], 0, 1])
					
					intersectionFound, intersectingPoint1, intersectingPoint2 = caller.intersectWithFinitePlane(
						slicePlaneNormal, slicePlaneOrigin, intersectingPlaneOrigin[0:3], intersectingPlaneX[0:3],
						intersectingPlaneY[0:3])
					
					if intersectionFound < 2:
						continue
					intersectingPoint1[2] = 0
					intersectingPoint2[2] = 0
					
					o1 = np.array(intersectingPoint1, dtype=float)
					o2 = np.array(intersectingPoint2, dtype=float)
					self.endpoints.append(o1)
					self.endpoints.append(o2)
				
		
	
	def __init__(self, parent=None):
		"""
		Called when the user opens the module the first time and the widget is initialized.
		"""
		ScriptedLoadableModuleWidget.__init__(self, parent)
		VTKObservationMixin.__init__(self)  # needed for parameter node observation
		self.logic = None
		self.intersectionVariables = self.IntersectionVariables()
		

	def setup(self):
		"""
		Called when the user opens the module the first time and the widget is initialized.
		"""
		ScriptedLoadableModuleWidget.setup(self)

		# Load widget from .ui file (created by Qt Designer).
		# Additional widgets can be instantiated manually and added to self.layout.
		uiWidget = slicer.util.loadUI(self.resourcePath('UI/IntersectionControls.ui'))
		self.layout.addWidget(uiWidget)
		self.ui = slicer.util.childWidgetVariables(uiWidget)

		# Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
		# "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
		# "setMRMLScene(vtkMRMLScene*)" slot.
		uiWidget.setMRMLScene(slicer.mrmlScene)

		# Create logic class. Logic implements all computations that should be possible to run
		# in batch mode, without a graphical user interface.
		self.logic = IntersectionControlsLogic()

		# Connections
		# vtk events that to be connected
		events = (
		  vtk.vtkCommand.LeftButtonPressEvent,
		  vtk.vtkCommand.LeftButtonReleaseEvent,
          vtk.vtkCommand.KeyPressEvent,
          vtk.vtkCommand.KeyReleaseEvent,
          vtk.vtkCommand.MouseMoveEvent,
          vtk.vtkCommand.EnterEvent,
          vtk.vtkCommand.LeaveEvent,
		)
		functions = {
			'Red': self._redProcessEvent,
			'Green': self._greenProcessEvent,
			'Yellow': self._yellowProcessEvent,
		}
		for c in self.logic.sliceColourToID.keys():
			interactor = slicer.app.layoutManager().sliceWidget(c).sliceView().interactor()
			for e in events:
				interactor.AddObserver(e, functions[c], 1.0)
		
				
	def processEvent(self, caller=None, event=None, colour=None):
		if colour is None or event is None or caller is None:
			return
		if colour not in self.logic.sliceColourToID.keys():
			return
		
		if event == "LeaveEvent":
			self.intersectionVariables.processInit()
		elif event == "EnterEvent":
			self.intersectionVariables.processInit(caller=self, colour=colour)
		elif event == "KeyPressEvent":
			if self.intersectionVariables.leftMousePressing:
				return
			key = caller.GetKeySym()
			if key == "Control_L" or key == "Alt_L" or key == "Shift_L":
				self.intersectionVariables.coreKeyPressing = True
		elif event == "KeyReleaseEvent":
			if self.intersectionVariables.leftMousePressing:
				return
			key = caller.GetKeySym()
			if key == "Control_L" or key == "Alt_L" or key == "Shift_L":
				self.intersectionVariables.coreKeyPressing = False
		elif event == "LeftButtonPressEvent":
			self.intersectionVariables.leftMousePressing = True
			if self.intersectionVariables.coreKeyPressing:
				return
			
			curPosXY = caller.GetEventPosition()
			curPosXYZ = np.array([curPosXY[0], curPosXY[1], 0], dtype=float)
			
			# Check Translate or rotate
			sliceNode = self.logic.sliceColourToNode[colour]
			m_XY_to_RAS = self.getNumpyMatrixFromVTK4x4(sliceNode.GetXYToRAS())
			m_RAS_to_XY = np.linalg.inv(m_XY_to_RAS)
			
			slicePlaneNormal = [0, 0, 1]
			slicePlaneOrigin = [0, 0, 0]
			
			# get colour closest to mouse
			colourSelected = None
			c_dist = 100000
			
			for c in self.logic.sliceColourToID.keys():
				if c == colour:
					continue
				intersectingSliceNode = self.logic.sliceColourToNode[c]
				intersectingSliceDimension = intersectingSliceNode.GetDimensions()
				intersecting_m_XY_to_RAS = self.getNumpyMatrixFromVTK4x4(intersectingSliceNode.GetXYToRAS())
				
				m_IntersectingXY_to_XY = np.matmul(m_RAS_to_XY, intersecting_m_XY_to_RAS)
				
				intersectingPlaneOrigin = np.matmul(m_IntersectingXY_to_XY, [0, 0, 0, 1])
				intersectingPlaneX = np.matmul(m_IntersectingXY_to_XY, [intersectingSliceDimension[0], 0, 0, 1])
				intersectingPlaneY = np.matmul(m_IntersectingXY_to_XY, [0, intersectingSliceDimension[1], 0, 1])
				
				intersectionFound, intersectingPoint1, intersectingPoint2 = self.intersectWithFinitePlane(
					slicePlaneNormal, slicePlaneOrigin, intersectingPlaneOrigin[0:3], intersectingPlaneX[0:3],
					intersectingPlaneY[0:3])
				
				if intersectionFound < 2:
					continue
				intersectingPoint1[2] = 0
				intersectingPoint2[2] = 0
				
				o1 = np.array(intersectingPoint1, dtype=float)
				d1 = np.array(intersectingPoint2, dtype=float) - o1
				d1 = d1 / np.linalg.norm(d1)
				
				o2 = np.array(intersectingPoint2, dtype=float)
				
				relativePoint = curPosXYZ - o1
				if min(np.linalg.norm(curPosXYZ - o1),np.linalg.norm(curPosXYZ - o2)) < 10:
					# Near an endpoint of a projection line, rotation mode on
					self.intersectionVariables.rotationMode = True
					colourSelected = None
					break
					
				distance = np.linalg.norm(relativePoint - np.dot(relativePoint, d1) * d1)
				if c_dist > distance and not self.intersectionVariables.rotationMode:
					c_dist = distance
					colourSelected = c
				
			if colourSelected is not None or c_dist < 10:
				# translation mode
				self.intersectionVariables.colourSelected = colourSelected
			
			if self.intersectionVariables.rotationMode:
				# rotation mode
				temp_rotationCenter = self.getIntersectionPoint(colour=colour)
				self.intersectionVariables.rotationCenter = np.array([temp_rotationCenter[0], temp_rotationCenter[1], 0, 1], dtype=float)
				self.intersectionVariables.rotationCenterRAS = np.matmul(
					self.getNumpyMatrixFromVTK4x4(self.logic.sliceColourToNode[colour].GetXYToRAS()),
					self.intersectionVariables.rotationCenter)
				self.intersectionVariables.rotationPrevAngleRad = self.getSliceRotationAngleRad(curPosXYZ=curPosXYZ)
		elif event == "LeftButtonReleaseEvent":
			self.intersectionVariables.processInit(caller=self, colour=colour)
		elif event == "MouseMoveEvent":
			curPosXY = caller.GetEventPosition()
			curPosXYZ = np.array([curPosXY[0], curPosXY[1], 0], dtype=float)
			self.mouseHoveringCheck(colour=colour, mouseXYZ=curPosXYZ)
			
			if not self.intersectionVariables.leftMousePressing:
				return
			if self.intersectionVariables.coreKeyPressing:
				return
			
			if self.intersectionVariables.rotationMode:
				# check all variables needed
				if self.intersectionVariables.rotationCenter is None or self.intersectionVariables.rotationCenterRAS is None or self.intersectionVariables.rotationPrevAngleRad is None:
					return
				self.processRotateProjectedSlices(colourBase=colour, mouseXYZ=curPosXYZ)
			else:
				if self.intersectionVariables.colourSelected is None:
					return
				if self.intersectionVariables.colourSelected == colour:
					return
				self.processTranslateProjectedSlice(colourBase=colour, mouseXYZ=curPosXYZ)

	
	def mouseHoveringCheck(self, colour=None, mouseXYZ=None):
		if colour is None or mouseXYZ is None:
			slicer.app.restoreOverrideCursor()
			return
		if self.intersectionVariables.endpoints is None:
			self.intersectionVariables.updateEndpoints(colour=colour, caller=self)
		isNearEndpoint = False
		for p in self.intersectionVariables.endpoints:
			if np.linalg.norm(mouseXYZ - p) < 10:
				isNearEndpoint = True
				break
		if isNearEndpoint:
			slicer.app.setOverrideCursor(qt.Qt.PointingHandCursor)
		else:
			slicer.app.restoreOverrideCursor()
		
	
	def processRotateProjectedSlices(self, colourBase, mouseXYZ):
		sliceRotationAngleRad = self.getSliceRotationAngleRad(curPosXYZ=mouseXYZ)
		sliceNode = self.logic.sliceColourToNode[colourBase]
		m_Slice_to_RAS = self.getNumpyMatrixFromVTK4x4(sliceNode.GetSliceToRAS())
		
		if vtk.vtkMath().Determinant3x3(m_Slice_to_RAS[0,0:3], m_Slice_to_RAS[1,0:3], m_Slice_to_RAS[2,0:3]) >= 0:
			rotateDirection = 1.0
		else:
			rotateDirection = -1.0
		
		rotatedSliceToSliceTransform = vtk.vtkTransform()
		rotatedSliceToSliceTransform.Translate(self.intersectionVariables.rotationCenterRAS[0], self.intersectionVariables.rotationCenterRAS[1],
		                                       self.intersectionVariables.rotationCenterRAS[2])
		rotatedSliceToSliceTransform.RotateWXYZ(
			rotateDirection * vtk.vtkMath().DegreesFromRadians(sliceRotationAngleRad - self.intersectionVariables.rotationPrevAngleRad),
			m_Slice_to_RAS[0,2],
			m_Slice_to_RAS[1,2],
			m_Slice_to_RAS[2,2])
		rotatedSliceToSliceTransform.Translate(-self.intersectionVariables.rotationCenterRAS[0], -self.intersectionVariables.rotationCenterRAS[1],
		                                       -self.intersectionVariables.rotationCenterRAS[2])
		
		self.intersectionVariables.rotationPrevAngleRad = sliceRotationAngleRad
		for c, projSliceNode in self.logic.sliceColourToNode.items():
			if c == colourBase:
				continue
			rotatedSliceToRAS = vtk.vtkMatrix4x4()
			vtk.vtkMatrix4x4().Multiply4x4(rotatedSliceToSliceTransform.GetMatrix(), projSliceNode.GetSliceToRAS(), rotatedSliceToRAS)
			projSliceNode.GetSliceToRAS().DeepCopy(rotatedSliceToRAS)
			projSliceNode.UpdateMatrices()
			
	
	def processTranslateProjectedSlice(self, colourBase, mouseXYZ):
		sliceNode = self.logic.sliceColourToNode[colourBase]
		m_XY_to_RAS = self.getNumpyMatrixFromVTK4x4(sliceNode.GetXYToRAS())
		m_RAS_to_XY = np.linalg.inv(m_XY_to_RAS)
		
		slicePlaneNormal = [0, 0, 1]
		slicePlaneOrigin = [0, 0, 0]
		
		intersectingSliceNode = self.logic.sliceColourToNode[self.intersectionVariables.colourSelected]
		intersectingSliceDimension = intersectingSliceNode.GetDimensions()
		intersecting_m_XY_to_RAS = self.getNumpyMatrixFromVTK4x4(intersectingSliceNode.GetXYToRAS())
		
		m_IntersectingXY_to_XY = np.matmul(m_RAS_to_XY, intersecting_m_XY_to_RAS)
		
		intersectingPlaneOrigin = np.matmul(m_IntersectingXY_to_XY, [0, 0, 0, 1])
		intersectingPlaneX = np.matmul(m_IntersectingXY_to_XY, [intersectingSliceDimension[0], 0, 0, 1])
		intersectingPlaneY = np.matmul(m_IntersectingXY_to_XY, [0, intersectingSliceDimension[1], 0, 1])
		
		intersectionFound, intersectingPoint1, intersectingPoint2 = self.intersectWithFinitePlane(
			slicePlaneNormal, slicePlaneOrigin, intersectingPlaneOrigin[0:3], intersectingPlaneX[0:3],
			intersectingPlaneY[0:3])
		
		if intersectionFound < 2:
			return
		intersectingPoint1[2] = 0
		intersectingPoint2[2] = 0
		
		# line = o + d * t
		o1 = np.array(intersectingPoint1, dtype=float)
		d1 = np.array(intersectingPoint2, dtype=float) - o1
		d1 = d1 / np.linalg.norm(d1)
		
		o2 = np.array(intersectingPoint2, dtype=float)
		d2 = np.array(intersectingPoint1, dtype=float) - o2
		d2 = d2 / np.linalg.norm(d2)
		
		originRAS = np.matmul(intersecting_m_XY_to_RAS,
		                      [intersectingSliceDimension[0] / 2, intersectingSliceDimension[1] / 2, 0, 1])
		
		projectedXYZ1 = np.append(mouseXYZ - np.dot(d1, mouseXYZ - o1) * d1, [1])
		projectedXYZ2 = np.append(mouseXYZ - np.dot(d2, mouseXYZ - o2) * d2, [1])
		middleXYZ = np.append((o1 + o2) / 2, [1])
		
		projectedRAS = np.matmul(m_XY_to_RAS, (projectedXYZ1 + projectedXYZ2) / 2)
		middleRAS = np.matmul(m_XY_to_RAS, middleXYZ)
		dRAS = projectedRAS - middleRAS
		
		newCenter = originRAS + dRAS
		intersectingSliceNode.JumpSliceByCentering(newCenter[0], newCenter[1], newCenter[2])
		intersectingSliceNode.UpdateMatrices()
		

	# Get Angle of the mouse respect to the intersect point in rad
	def getSliceRotationAngleRad(self, curPosXYZ):
		if self.intersectionVariables.rotationCenter is None:
			return None
		import math
		return math.atan2(curPosXYZ[1] - self.intersectionVariables.rotationCenter[1], curPosXYZ[0] - self.intersectionVariables.rotationCenter[0])
	
	
	# Get intersection point of two projected lines
	def getIntersectionPoint(self, colour):
		if colour not in self.logic.sliceColourToID.keys():
			return None
		projectionColours = list(self.logic.sliceColourToID.keys())
		projectionColours.remove(colour)
		
		success1, o1, d1 = self.getParaLine(colourBase=colour, colourProj=projectionColours[0])
		success2, o2, d2 = self.getParaLine(colourBase=colour, colourProj=projectionColours[1])
		if not (success2 and success1):
			return None
		t = (d2[1]*(o2[0]-o1[0]) - d2[0]*(o2[1]-o1[1])) / (d1[0]*d2[1]-d2[0]*d1[1])
		return o1+d1*t
		
	
	# Get parametric line of colourProj on the XY coordinate system of colourBase
	def getParaLine(self, colourBase, colourProj, interchange=False):
		if colourBase not in self.logic.sliceColourToID.keys() or colourProj not in self.logic.sliceColourToID.keys():
			return False, None, None
		
		sliceNode = self.logic.sliceColourToNode[colourBase]
		m_XY_to_RAS = self.getNumpyMatrixFromVTK4x4(sliceNode.GetXYToRAS())
		m_RAS_to_XY = np.linalg.inv(m_XY_to_RAS)
		
		slicePlaneNormal = [0, 0, 1]
		slicePlaneOrigin = [0, 0, 0]
		
		intersectingSliceNode = self.logic.sliceColourToNode[colourProj]
		intersectingSliceDimension = intersectingSliceNode.GetDimensions()
		intersecting_m_XY_to_RAS = self.getNumpyMatrixFromVTK4x4(intersectingSliceNode.GetXYToRAS())
		
		m_IntersectingXY_to_XY = np.matmul(m_RAS_to_XY, intersecting_m_XY_to_RAS)
		
		intersectingPlaneOrigin = np.matmul(m_IntersectingXY_to_XY, [0, 0, 0, 1])
		intersectingPlaneX = np.matmul(m_IntersectingXY_to_XY, [intersectingSliceDimension[0], 0, 0, 1])
		intersectingPlaneY = np.matmul(m_IntersectingXY_to_XY, [0, intersectingSliceDimension[1], 0, 1])
		
		intersectionFound, intersectingPoint1, intersectingPoint2 = self.intersectWithFinitePlane(
			slicePlaneNormal, slicePlaneOrigin, intersectingPlaneOrigin[0:3], intersectingPlaneX[0:3],
			intersectingPlaneY[0:3])
		if intersectionFound < 2:
			return False, None, None
		if interchange:
			o = np.array(intersectingPoint2[0:2], dtype=float)
			d = np.array(intersectingPoint1[0:2], dtype=float) - o
			d = d / np.linalg.norm(d)
		else:
			o = np.array(intersectingPoint1[0:2], dtype=float)
			d = np.array(intersectingPoint2[0:2], dtype=float) - o
			d = d / np.linalg.norm(d)
		return True, o, d
		
	
	# Get number of intersections and two points of intersections between an infinite plane and finite
	def intersectWithFinitePlane(self, n, o, pOrigin, px, py):
		numIntersections = 0
		intersectingPoint1 = [0, 0, 1]
		intersectingPoint2 = [0, 0, 1]
		
		x = [0,0,1]
		t = vtk.mutable(0)
		xr0 = pOrigin.copy()
		xr1 = px.copy()
		if vtk.vtkPlane().IntersectWithLine(xr0, xr1, n, o, t, x):
			intersectingPoint1 = x.copy()
			numIntersections += 1
			
		xr1 = py.copy()
		if vtk.vtkPlane().IntersectWithLine(xr0, xr1, n, o, t, x):
			if numIntersections == 0:
				intersectingPoint1 = x.copy()
			else:
				intersectingPoint2 = x.copy()
			numIntersections += 1
		if numIntersections == 2:
			return numIntersections, intersectingPoint1, intersectingPoint2
		
		xr0 = -pOrigin + px + py
		if vtk.vtkPlane().IntersectWithLine(xr0, xr1, n, o, t, x):
			if numIntersections == 0:
				intersectingPoint1 = x.copy()
			else:
				intersectingPoint2 = x.copy()
			numIntersections += 1
		if numIntersections == 2:
			return numIntersections, intersectingPoint1, intersectingPoint2
		
		xr1 = px.copy()
		if vtk.vtkPlane().IntersectWithLine(xr0, xr1, n, o, t, x):
			if numIntersections == 0:
				intersectingPoint1 = x.copy()
			else:
				intersectingPoint2 = x.copy()
			numIntersections += 1
		return numIntersections, intersectingPoint1, intersectingPoint2
		
		
	def getNumpyMatrixFromVTK4x4(self, vtk4x4):
		rtn = np.ones((4,4))
		for i in range(4):
			for j in range(4):
				rtn[i,j] = vtk4x4.GetElement(i, j)
		return rtn

	def _redProcessEvent(self, caller=None, event=None):
		self.processEvent(caller=caller, event=event, colour='Red')
	
	def _greenProcessEvent(self, caller=None, event=None):
		self.processEvent(caller=caller, event=event, colour='Green')
	
	def _yellowProcessEvent(self, caller=None, event=None):
		self.processEvent(caller=caller, event=event, colour='Yellow')

	def cleanup(self):
		"""
		Called when the application closes and the module widget is destroyed.
		"""
		self.removeObservers()

	def enter(self):
		"""
		Called each time the user opens this module.
		"""
		viewNodes = slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode")
		for viewNode in viewNodes:
			viewNode.SetSliceIntersectionVisibility(1)

	def exit(self):
		"""
		Called each time the user opens a different module.
		"""
		pass

	def onSceneStartClose(self, caller, event):
		"""
		Called just before the scene is closed.
		"""
		pass

	def onSceneEndClose(self, caller, event):
		"""
		Called just after the scene is closed.
		"""
		pass

	def setParameterNode(self, inputParameterNode):
		"""
		Set and observe parameter node.
		Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
		"""
		pass


#
# IntersectionControlsLogic
#

class IntersectionControlsLogic(ScriptedLoadableModuleLogic):
	"""This class should implement all the actual
	computation done by your module.  The interface
	should be such that other python code can import
	this class and make use of the functionality without
	requiring an instance of the Widget.
	Uses ScriptedLoadableModuleLogic base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self):
		"""
		Called when the logic class is instantiated. Can be used for initializing member variables.
		"""
		ScriptedLoadableModuleLogic.__init__(self)
		self.sliceColourToID = {
			'Red': 'vtkMRMLSliceNodeRed',
			'Green': 'vtkMRMLSliceNodeGreen',
			'Yellow': 'vtkMRMLSliceNodeYellow',
		}
		self.sliceColourToNode = {c: slicer.util.getNode(nodeID) for c, nodeID in self.sliceColourToID.items()}

	def setDefaultParameters(self, parameterNode):
		"""
		Initialize parameter node with default settings.
		"""
		pass


#
# IntersectionControlsTest
#

class IntersectionControlsTest(ScriptedLoadableModuleTest):
	"""
	This is the test case for your scripted module.
	Uses ScriptedLoadableModuleTest base class, available at:
	https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def setUp(self):
		""" Do whatever is needed to reset the state - typically a scene clear will be enough.
		"""
		slicer.mrmlScene.Clear()

	def runTest(self):
		"""Run as few or as many tests as needed here.
		"""
		pass

	def test_IntersectionControls1(self):
		""" Ideally you should have several levels of tests.  At the lowest level
		tests should exercise the functionality of the logic with different inputs
		(both valid and invalid).  At higher levels your tests should emulate the
		way the user would interact with your code and confirm that it still works
		the way you intended.
		One of the most important features of the tests is that it should alert other
		developers when their changes will have an impact on the behavior of your
		module.  For example, if a developer removes a feature that you depend on,
		your test should break so they know that the feature is needed.
		"""
		pass
