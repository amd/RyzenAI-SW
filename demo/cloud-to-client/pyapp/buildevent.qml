import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick.Controls.Basic


Window {
    id:mainwindow
    width: 1280
    height: 720
    title: qsTr("AMD Image Categorizer Demo")
    visible: true
    color: "#181818"
    modality: Qt.ApplicationModal
    flags:Qt.FramelessWindowHint | Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint
    property bool isMaximized :false
    property int mainwindowmargin: mainwindow.width * 0.0390625
    property string searchCategory:""
    property bool isCurrentScanWithRyzenAI:false
    property bool isCurrentScanWithAzureEP:false
    property string msgDialogText: "Please select image Source Location folder using 'Browse' button to Scan with Ryzen AI/ Scan with Azure EP"


    // Title Bar
    Rectangle {
        id: titleBar
        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
        }
        z:3
        height: 32
        color: "#282828"
        MouseArea{
            id: iMouseArea
            property int prevX: 0
            property int prevY: 0
            anchors.fill: parent
            onPressed: {
                prevX=mouseX; prevY=mouseY}
            onPositionChanged:{
                var deltaX = mouseX - prevX;
                mainwindow.x += deltaX;
                prevX = mouseX - deltaX;

                var deltaY = mouseY - prevY
                mainwindow.y += deltaY;
                prevY = mouseY - deltaY;
            }
        }

        Button
        {
            id: logoButton
            anchors {
                left: parent.left
                leftMargin: 0
                verticalCenter: parent.verticalCenter
            }
            opacity: 1
            flat: true
            activeFocusOnTab: false
            icon.source:  "./images/icon_16_amd_icon@2x.png"
            icon.width: 32
            icon.height:32
            icon.color : "white"
            background: Rectangle {
                implicitWidth:32
                implicitHeight: 32
                color: "#000000" //"#DD0333"
            }
            width: 32
            height:  32

            ToolTip
            {
                delay: Qt.styleHints.mousePressAndHoldInterval
                visible: logoButton.hovered
                contentItem: Text {
                    id: txtToolTipLogoButton
                    text:"Search and Sort the images on your PC into different categories for a quick and easy viewing experience. Use the power of Ryzen AI to search the images locally when the cloud functionality is not available due to lost or no connectivity. The combination for Ryzen AI and Cloud compute ensures this feature is always available to the user."
                    font.family:"Segoe UI"
                    font.pixelSize: 14
                    wrapMode: Text.RichText
                    color: "#C0C0C0"
                }
                background: Rectangle {
                    radius: 4
                    color: "#303030"
                }
            }
        }




        // Transparent rectangle with app text
        Rectangle {
            id: titleTextRect
            anchors {
                verticalCenter: parent.verticalCenter
                left: logoButton.right
                leftMargin: 4
            }
            height: 32
            width: 185
            color: "transparent"
            opacity: 1
            // App text
            Text
            {
                anchors.fill: parent
                id: titleText
                text: "AMD Image Categorizer"
                textFormat: Text.RichText
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment :Text.AlignLeft
                font.bold: false
                font.pixelSize: 16
                color: "#FFFFFF"
            }
        }


        Row
        {
            id: textButtonRow
            spacing: 0
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.verticalCenter: parent.verticalCenter
            height: 32

            Button
            {
                id: minButton
                height: 32
                width: 32
                icon.source:  "./images/icon_16_minimize@2x.png"
                icon.color:  "#FFFFFF"
                icon.width: 16
                icon.height:16
                flat: true
                opacity: 1
                onClicked:
                {
                    showMinimized()
                }

                ToolTip
                {
                    delay: Qt.styleHints.mousePressAndHoldInterval
                    visible: minButton.hovered
                    contentItem: Text {
                        id: txtToolTipMinimize
                        text:qsTr("Minimize")
                        font.pixelSize: 14
                        wrapMode: Text.RichText
                        color: "#C0C0C0"
                        font.family: "Segoe UI"

                    }
                    background: Rectangle {
                        radius: 4
                        color: "#303030"
                    }
                }

                background: Rectangle {
                    implicitWidth: 32
                    implicitHeight:  32
                    radius: 0
                    border.color:"#00000080"
                    color: minButton.hovered? "#686868"  : "#00000080"

                }
            }

            //Maximize
            Button
            {
                id: maxButton
                height: 32
                width: 32
                icon.source: isMaximized ? "./images/restore-icon.png" : "./images/maximize-icon.png"
                icon.color:  "#FFFFFF"
                icon.width: 16
                icon.height:16
                flat: true
                opacity: 1
                onClicked:
                {
                    if(isMaximized)
                    {
                        isMaximized = false
                        showNormal()
                    }else
                    {
                        isMaximized = true
                        showMaximized()
                    }
                }

                ToolTip
                {
                    delay: Qt.styleHints.mousePressAndHoldInterval
                    visible: maxButton.hovered
                    contentItem: Text {
                        id: txtToolTipMaximize
                        text:  isMaximized? qsTr("Restore") : qsTr("Maximize")
                        font.pixelSize: 14
                        wrapMode: Text.RichText
                        color: "#C0C0C0"
                        font.family: "Segoe UI"

                    }
                    background: Rectangle {
                        radius: 4
                        color: "#303030"
                    }
                }

                background: Rectangle {
                    implicitWidth: 32
                    implicitHeight:  32
                    radius: 0
                    border.color:"#00000080"
                    color: maxButton.hovered? "#686868"  : "#00000080"

                }
            }

            // Close button: rightmost
            Button
            {
                id: closeButton
                height: 32
                width: 32
                icon.source:  "./images/icon_16_close@2x.png"
                icon.color:  "#FFFFFF"
                icon.width: 32
                icon.height:32
                //                topPadding: -2
                flat: true
                opacity: 1

                ToolTip
                {
                    delay: Qt.styleHints.mousePressAndHoldInterval
                    visible: closeButton.hovered
                    contentItem: Text {
                        id: txtToolTipClose
                        text:qsTr("Close")
                        font.family: "Segoe UI"
                        font.pixelSize: 14
                        wrapMode: Text.RichText
                        color: "#C0C0C0"

                    }
                    background: Rectangle {
                        radius: 4
                        color: "#303030"
                    }
                }

                onClicked:
                {
                    close()
                }

                background: Rectangle {
                    implicitWidth: 32
                    implicitHeight: 32
                    radius: 0
                    border.color:"#00000080"
                    color: closeButton.hovered? "#DD0333"  : "#00000080"

                }

            }


        }
    }// title bar





    //Content Area
    Item {
        id: contentArea

        anchors {
            top: titleBar.bottom
            left: parent.left
            right: parent.right
            //bottom: parent.bottom
        }


        Flickable {
            id: contentAreaColumnflick
            contentHeight: contentAreaColumn.height
            clip: true
            height: mainwindow.height - titleBar.height
            width: contentArea.width

            ScrollBar.vertical: ScrollBar {
                id:contentAreaColumnflickcrollbar
                parent: contentAreaColumnflick
                height: contentAreaColumnflick.availableHeight
                interactive: true
                width: 12
                anchors.right: contentAreaColumnflick.right
                implicitWidth:12
                //policy: ScrollBar.AsNeeded
            }

            ColumnLayout
            {
                id: contentAreaColumn
                anchors {
                    top: parent.top
                    left: parent.left
                    right: parent.right
                    //bottom: parent.bottom
                }
                anchors.topMargin:  20
                // anchors.bottomMargin:  20

                anchors.leftMargin: mainwindowmargin//100
                anchors.rightMargin: mainwindowmargin//100

                Text {
                    id:txtDesc
                    text: "Search and Sort the images on your PC into different categories for a quick and easy viewing experience. Use the power of Ryzen AI to search the images locally when the cloud functionality is not available due to lost or no connectivity. The combination for Ryzen AI and Cloud compute ensures this feature is always available to the user."
                    textFormat: Text.StyledText
                    style: Text.Raised
                    wrapMode: Text.WordWrap
                    height: 60
                    font.pixelSize:18
                    font.family: "Segoe UI"
                    horizontalAlignment :Text.AlignLeft
                    verticalAlignment: Text.AlignVCenter
                    color: "#E0E0E0"
                    Layout.fillWidth: true
                    width: (mainwindow.width - (mainwindowmargin*2))

                }



                Label {
                    id:lblImgLocation
                    text: qsTr("Image Source Location - Select Images directory using 'Browse' button")
                    textFormat: Text.RichText
                    height: 20
                    font.pixelSize:18
                    font.family: "Segoe UI"
                    horizontalAlignment :Text.AlignLeft
                    verticalAlignment: Text.AlignVCenter
                    color: "#FFFFFF"
                    Layout.topMargin: 12

                }

                Rectangle
                {
                    id: rect
                    Layout.topMargin: 0
                    color: "#202020"
                    border.color: "#202020"

                    width: (mainwindow.width - (mainwindowmargin*2))
                    Layout.preferredWidth: (mainwindow.width - (mainwindowmargin*2))
                    height: 84 - ( processStatusResult.visible  ? 0 : (processingProgressFaceDetect.visible ? -16 : 16))
                    Layout.preferredHeight:84 - ( processStatusResult.visible  ? 0 : (processingProgressFaceDetect.visible ? -16 : 16))
                    radius: 4


                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight:  true
                        spacing: 0

                        RowLayout
                        {
                            Layout.fillWidth: true
                            Layout.fillHeight:  true

                            spacing:  mainwindow.width*0.00625//8

                            TextField {
                                id:txtInputLocationPath
                                Layout.preferredHeight: 28
                                Layout.preferredWidth:  mainwindow.width*0.57 - (busyIndication.visible ? (busyIndication.width+mainwindow.width*0.00625) : 0 ) + (browseBtn.visible ? 0 : (browseBtn.width + mainwindow.width*0.00625) )+ (scanWithoutIPUBtn.visible ? 0 : (scanWithoutIPUBtn.width + mainwindow.width*0.00625)) + (startScanBtn.visible ? 0 : (startScanBtn.width +mainwindow.width*0.00625) )
                                width:   mainwindow.width*0.57 - (busyIndication.visible ? (busyIndication.width+mainwindow.width*0.00625) : 0 ) + (browseBtn.visible ? 0 : (browseBtn.width + mainwindow.width*0.00625) )+ (scanWithoutIPUBtn.visible ? 0 : (scanWithoutIPUBtn.width + mainwindow.width*0.00625)) + (startScanBtn.visible ? 0 : (startScanBtn.width +mainwindow.width*0.00625) )
                                text: backend.getDirectory(folderDialog.folder)
                                placeholderText : "Browse for images"
                                placeholderTextColor : "gray"

                                color:"#FFFFFF"
                                font.family:"Segoe UI"
                                font.pixelSize: 14
                                readOnly:true
                                background: Rectangle {
                                    radius: 3
                                    border.color: "#404040"
                                    color: "#282828"
                                    border.width: 1
                                }
                                Layout.topMargin: 16
                                Layout.leftMargin:  mainwindow.width*0.025//32
                            }

                            //progress bar
                            ProgressBar {
                                id: processingProgress
                                value: ((backend.currentProcessedCount/ backend.totalCount)*100)/100
                                visible:false
                                Layout.topMargin: 16
                                height:28
                                Layout.preferredHeight: 28
                                Layout.preferredWidth:  mainwindow.width*0.57 - (busyIndication.visible ? (busyIndication.width+mainwindow.width*0.00625) : 0 ) + (browseBtn.visible ? 0 : (browseBtn.width + mainwindow.width*0.00625) )+ (scanWithoutIPUBtn.visible ? 0 : (scanWithoutIPUBtn.width + mainwindow.width*0.00625)) + (startScanBtn.visible ? 0 : (startScanBtn.width +mainwindow.width*0.00625) )
                                width:   mainwindow.width*0.57 - (busyIndication.visible ? (busyIndication.width+mainwindow.width*0.00625) : 0 ) + (browseBtn.visible ? 0 : (browseBtn.width + mainwindow.width*0.00625) )+ (scanWithoutIPUBtn.visible ? 0 : (scanWithoutIPUBtn.width + mainwindow.width*0.00625)) + (startScanBtn.visible ? 0 : (startScanBtn.width +mainwindow.width*0.00625) )
                                Layout.leftMargin:  mainwindow.width*0.025//32

                                background: Rectangle {
                                    implicitWidth: mainwindow.width*0.4640625
                                    implicitHeight: 28
                                    color: "#404040"
                                    radius: 4
                                }

                                contentItem: Item {
                                    implicitWidth: mainwindow.width*0.4640625
                                    implicitHeight: 28
                                    Rectangle {
                                        width: processingProgress.visualPosition * parent.width
                                        height: parent.height
                                        radius: 2
                                        color: "#228b22"//"#DD0333"
                                    }
                                    Text {
                                        id: txtProgressBar
                                        text:  qsTr("Object Detection \t Processing %1 out of %2 images...").arg(backend.currentProcessedCount).arg(backend.totalCount)
                                        font.family:"Segoe UI"
                                        font.pixelSize: 14
                                        wrapMode: Text.RichText
                                        color: "#FFFFFF"
                                        leftPadding:8
                                        topPadding:4
                                    }

                                }

                            }


                            FolderDialog {
                                id: folderDialog
                                currentFolder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0]
                                //folder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0]
                            }


                            //Browse Button
                            Button {
                                id:browseBtn
                                text : qsTr("Browse")
                                Layout.alignment: Qt.AlignRight
                                font.family:"Segoe UI"
                                font.pixelSize: 14
                                width: mainwindow.width*0.0609375//78
                                height:  28
                                flat: true
                                visible: true
                                Layout.topMargin: 16

                                /*
                        ToolTip
                        {
                            delay: Qt.styleHints.mousePressAndHoldInterval
                            visible: browseBtn.hovered
                            contentItem: Text {
                                id: txtToolTipBrowseButton
                                text:qsTr("Browse")
                                font.family:"Segoe UI"
                                font.pixelSize: 14
                                wrapMode: Text.RichText
                                color: "#C0C0C0"
                            }
                            background: Rectangle {
                                radius: 4
                                color: "#303030"
                            }
                        }
                        */

                                contentItem: Text {
                                    text: browseBtn.text
                                    font: browseBtn.font
                                    topPadding:-2
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    color: "#FFFFFF"
                                }

                                onClicked:
                                {
                                    folderDialog.open()
                                }

                                background: Rectangle {
                                    id:rectbrowseBtn
                                    implicitWidth:  mainwindow.width*0.0609375//78
                                    implicitHeight: 28
                                    width:   mainwindow.width*0.0609375//78
                                    height:  28
                                    border.width:1
                                    radius: 1
                                    border.color: "#00000040"
                                    color: browseBtn.pressed? "#228b22" : ( browseBtn.hovered ?  "#686868" : "#404040")// browseBtn.hovered ? "#686868" : "#404040"
                                }

                            }

                            //Spinner
                            BusyIndicator {
                                id: busyIndication
                                visible: false
                                palette.dark: "#FFFFFF"
                                height:28
                                width:28
                                implicitHeight : 28
                                implicitWidth:28
                                Layout.topMargin:16

                                layer.smooth :true
                                contentItem: Item {
                                    implicitWidth: 28
                                    implicitHeight: 28

                                    Item {
                                        id: item
                                        x: parent.width / 2 - 14
                                        y: parent.height / 2 - 14
                                        width: 28
                                        height: 28
                                        opacity: busyIndication.running ? 1 : 0

                                        Behavior on opacity {
                                            OpacityAnimator {
                                                duration: 250
                                            }
                                        }

                                        RotationAnimator {
                                            target: item
                                            running: busyIndication.visible && busyIndication.running
                                            from: 0
                                            to: 360
                                            loops: Animation.Infinite
                                            duration:2000
                                        }

                                        Repeater {
                                            id: repeater
                                            model: 8

                                            Rectangle {
                                                x: item.width / 2 - width / 2
                                                y: item.height / 2 - height / 2
                                                implicitWidth: 5
                                                implicitHeight: 5
                                                radius: 2
                                                color: "#FFFFFF"
                                                transform: [
                                                    Translate {
                                                        y: -Math.min(item.width, item.height) * 0.5 + 5
                                                    },
                                                    Rotation {
                                                        angle: index / repeater.count * 360
                                                        origin.x: 2.5
                                                        origin.y: 2.5
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                }
                            }

                            //warning dialog
                            Window {
                                id: msgDialog
                                modality: Qt.ApplicationModal
                                title: "Warning"
                                visible: false
                                flags: Qt.FramelessWindowHint | Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint
                                minimumHeight: 90
                                minimumWidth: 500
                                color: "#404040"
                                Label {
                                    anchors.margins: 10
                                    anchors.top: parent.top
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.bottom: messageBoxButton.top
                                    horizontalAlignment: Text.AlignHCenter
                                    wrapMode: Text.WordWrap
                                    id: messageBoxLabel
                                    color:"#FFFFFF"
                                    text:  msgDialogText
                                }

                                Button {
                                    anchors.margins: 10
                                    id: messageBoxButton
                                    height:28
                                    width:50
                                    anchors.bottom: parent.bottom
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "Ok"
                                    onClicked: msgDialog.visible = false
                                }
                            }

                            //start Scan Button
                            Button {
                                id:startScanBtn
                                text : "Scan with Ryzen AI"
                                Layout.alignment: Qt.AlignRight
                                font.family:"Segoe UI"
                                font.pixelSize: 14
                                width:  mainwindow.width*0.1125//mainwindow.width*0.075//96
                                height:  28
                                flat: true
                                visible: true
                                Layout.topMargin: 16
                                /*
                                ToolTip
                                {
                                    delay: Qt.styleHints.mousePressAndHoldInterval
                                    visible: startScanBtn.hovered
                                    contentItem: Text {
                                        id: txtToolTipStartScanBtn
                                        text: startScanBtnText.text === "Stop" ? "Stop Scan" : "Scan with Ryzen AI"
                                        font.family:"Segoe UI"
                                        font.pixelSize: 14
                                        wrapMode: Text.RichText
                                        color: "#C0C0C0"
                                    }
                                    background: Rectangle {
                                        radius: 4
                                        color: "#303030"
                                    }
                                }
                                */

                                contentItem: Text {
                                    id:startScanBtnText
                                    text: startScanBtn.text
                                    font: startScanBtn.font
                                    topPadding:-2
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    color: "#FFFFFF"
                                }

                                onClicked:
                                {
                                    if(txtInputLocationPath.text === "" )
                                    {
                                        msgDialogText = "Please select image source location folder using 'Browse' button to scan with Ryzen AI.\nNote: Currently supported image types are jpg, png, jpeg, bmp, tiff, webp, and tif."
                                        msgDialog.visible = true
                                    }else if(backend.getImageCount(folderDialog.folder) == 0)
                                    {
                                        msgDialogText = "The selected directory does not have any images. Please check.\nNote: Currently supported image types are jpg, png, jpeg, bmp, tiff, webp, and tif."
                                        msgDialog.visible = true
                                    }
                                    else
                                    {
                                        isCurrentScanWithRyzenAI = true
                                        isCurrentScanWithAzureEP=false


                                        if(startScanBtnText.text === "Stop")
                                        {
                                            busyIndication.visible = false
                                            processingProgress.visible = false

                                            startScanWithoutIPUBtnText.text = "Scan with Azure EP"
                                            startScanBtnText.text = "Scan with Ryzen AI"

                                            txtInputLocationPath.visible = true
                                            browseBtn.visible = true
                                            startScanBtn.visible = true
                                            scanWithoutIPUBtn.visible = true
                                            processStatusResult.visible =true

                                            lblCategories.visible = false
                                            txtMaxImageCount.visible = false
                                            txtMinImageCount.visible = false
                                            lblResults.visible = false
                                            rectSearch.visible = false
                                            rectHist.visible = false


                                            backend.stopScan(folderDialog.folder)
                                        }else
                                        {

                                            txtInputLocationPath.visible = false
                                            browseBtn.visible = false
                                            scanWithoutIPUBtn.visible = false
                                            processStatusResult.visible =false

                                            busyIndication.visible = true
                                            processingProgress.visible = true

                                            startScanBtnText.text = "Stop"

                                            lblCategories.visible = false
                                            txtMaxImageCount.visible = false
                                            txtMinImageCount.visible = false
                                            rectHist.visible = false
                                            lblResults.visible = false
                                            rectSearch.visible = false



                                            backend.startScan(folderDialog.folder)
                                        }
                                    }
                                }

                                background: Rectangle {
                                    id:rectStartScan
                                    implicitWidth: mainwindow.width*0.1125//96
                                    implicitHeight: 28
                                    width:  mainwindow.width*0.1125//96
                                    height:  28
                                    border.width:1
                                    radius: 1
                                    border.color: "#00000040"
                                    color: ((isCurrentScanWithRyzenAI) && startScanBtnText.text !== "Stop") ? "#006400" : ((startScanBtn.pressed ) && startScanBtnText.text !== "Stop") ? "#228b22" : ( startScanBtn.hovered ?  "#686868" : "#404040")
                                }

                            }

                            // Scan  Without IPU Button
                            Button {
                                id:scanWithoutIPUBtn
                                text : "Scan with Azure EP"
                                Layout.alignment: Qt.AlignRight
                                font.family:"Segoe UI"
                                font.pixelSize: 14
                                width: mainwindow.width*0.1125//144
                                height:  28
                                flat: true
                                visible: true
                                Layout.topMargin: 16
                                Layout.rightMargin: mainwindow.width*0.025//32

                                /*
                        ToolTip
                        {
                            delay: Qt.styleHints.mousePressAndHoldInterval
                            visible: scanWithoutIPUBtn.hovered
                            contentItem: Text {
                                id: txtToolTipScanWithoutIPUBtn
                                text: startScanWithoutIPUBtnText.text === "Stop" ? "Stop Scan" : "Scan with Azure EP"
                                font.family:"Segoe UI"
                                font.pixelSize: 14
                                wrapMode: Text.RichText
                                color: "#C0C0C0"
                            }
                            background: Rectangle {
                                radius: 4
                                color: "#303030"
                            }
                        }
                        */

                                contentItem: Text {
                                    id: startScanWithoutIPUBtnText
                                    text: scanWithoutIPUBtn.text
                                    font: scanWithoutIPUBtn.font
                                    topPadding:-2
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    color: "#FFFFFF"
                                }

                                onClicked:
                                {


                                    if(txtInputLocationPath.text === "" )
                                    {
                                        msgDialogText =  "Please select image source location folder using 'Browse' button to scan with Azure EP.\nNote: Currently supported image types are jpg, png, jpeg, bmp, tiff, webp, and tif."
                                        msgDialog.visible = true
                                    }else if(backend.getImageCount(folderDialog.folder) == 0)
                                    {
                                        msgDialogText = "The selected directory does not have any images. Please check.\nNote: Currently supported image types are jpg, png, jpeg, bmp, tiff, webp, and tif."
                                        msgDialog.visible = true
                                    }
                                    else if(backend.checkInternetConnection() === false )
                                    {
                                        msgDialogText = "Please check internet connection.\nNote: Internet connection is required to perform 'Scan with Azure EP'"
                                        msgDialog.visible = true
                                    }
                                    else
                                    {
                                        isCurrentScanWithRyzenAI = false
                                        isCurrentScanWithAzureEP =true


                                        if(startScanWithoutIPUBtnText.text === "Stop")
                                        {

                                            busyIndication.visible = false
                                            processingProgress.visible = false

                                            startScanWithoutIPUBtnText.text = "Scan with Azure EP"
                                            startScanBtnText.text = "Scan with Ryzen AI"

                                            txtInputLocationPath.visible = true
                                            browseBtn.visible = true
                                            startScanBtn.visible = true
                                            scanWithoutIPUBtn.visible = true
                                            processStatusResult.visible =true

                                            lblCategories.visible = false
                                            txtMaxImageCount.visible = false
                                            txtMinImageCount.visible = false
                                            lblResults.visible = false
                                            rectSearch.visible = false
                                            rectHist.visible = false

                                            backend.stopScanWithoutIPU(folderDialog.folder)
                                        }else
                                        {
                                            txtInputLocationPath.visible = false
                                            browseBtn.visible = false
                                            startScanBtn.visible = false
                                            processStatusResult.visible =false

                                            busyIndication.visible = true
                                            processingProgress.visible = true

                                            startScanWithoutIPUBtnText.text = "Stop"

                                            lblCategories.visible = false
                                            txtMaxImageCount.visible = false
                                            txtMinImageCount.visible = false
                                            rectHist.visible = false
                                            lblResults.visible = false
                                            rectSearch.visible = false



                                            backend.startScanWithoutIPU(folderDialog.folder)
                                        }
                                    }
                                }

                                background: Rectangle {
                                    id:rectScanWithoutIPUBtn
                                    implicitWidth:  mainwindow.width*0.1125//144
                                    implicitHeight: 28
                                    width:   mainwindow.width*0.1125//144
                                    height:  28
                                    border.width:1
                                    radius: 1
                                    border.color: "#00000040"
                                    color: ( (isCurrentScanWithAzureEP) && startScanWithoutIPUBtnText.text !== "Stop")? "#006400" : ( (scanWithoutIPUBtn.pressed ) && startScanWithoutIPUBtnText.text !== "Stop")? "#228b22" : ( scanWithoutIPUBtn.hovered ?  "#686868" : "#404040")
                                }

                            }


                        }


                        //progress bar face detection
                        ProgressBar {
                            id: processingProgressFaceDetect
                            value: ((backend.currentProcessedFaceDetectCount/ backend.totalCount)*100)/100
                            visible:processingProgress.visible
                            Layout.topMargin: 4
                            height:28
                            Layout.preferredHeight: 28
                            Layout.preferredWidth:  mainwindow.width*0.57 - (busyIndication.visible ? (busyIndication.width+mainwindow.width*0.00625) : 0 ) + (browseBtn.visible ? 0 : (browseBtn.width + mainwindow.width*0.00625) )+ (scanWithoutIPUBtn.visible ? 0 : (scanWithoutIPUBtn.width + mainwindow.width*0.00625)) + (startScanBtn.visible ? 0 : (startScanBtn.width +mainwindow.width*0.00625) )
                            width:   mainwindow.width*0.57 - (busyIndication.visible ? (busyIndication.width+mainwindow.width*0.00625) : 0 ) + (browseBtn.visible ? 0 : (browseBtn.width + mainwindow.width*0.00625) )+ (scanWithoutIPUBtn.visible ? 0 : (scanWithoutIPUBtn.width + mainwindow.width*0.00625)) + (startScanBtn.visible ? 0 : (startScanBtn.width +mainwindow.width*0.00625) )
                            Layout.leftMargin:  mainwindow.width*0.025//32

                            background: Rectangle {
                                implicitWidth: mainwindow.width*0.4640625
                                implicitHeight: 28
                                color: "#404040"
                                radius: 4
                            }

                            contentItem: Item {
                                implicitWidth: mainwindow.width*0.4640625
                                implicitHeight: 28
                                Rectangle {
                                    width: processingProgressFaceDetect.visualPosition * parent.width
                                    height: parent.height
                                    radius: 2
                                    color: "#228b22"//"#DD0333"
                                }
                                Text {
                                    id: txtProgressBarFaceDetect
                                    text:  qsTr("Face Detection \t Processing %1 out of %2 images...").arg(backend.currentProcessedFaceDetectCount).arg(backend.totalCount)
                                    font.family:"Segoe UI"
                                    font.pixelSize: 14
                                    wrapMode: Text.RichText
                                    color: "#FFFFFF"
                                    leftPadding:8
                                    topPadding:4
                                }

                            }

                        }

                    }

                    RowLayout
                    {
                        id:processStatusResult
                        Layout.fillWidth: true
                        Layout.fillHeight:  true
                        Layout.topMargin: 12
                        //Layout.bottomMargin: 16
                        spacing:  mainwindow.width*0.00625
                        Layout.alignment: Qt.AlignBottom
                        visible: false

                        Image {
                            id:imgCheckmark
                            source: "./images/checkmark.png"
                            width: processStatusResult.visible ?  16 : 0
                            height: processStatusResult.visible ? 16 : 0
                            Layout.minimumHeight:  processStatusResult.visible ? 16 : 0
                            Layout.minimumWidth: processStatusResult.visible ? 16 : 0
                            Layout.alignment:Qt.AlignLeft
                            Layout.topMargin:  processStatusResult.visible ? 52 : 0
                            Layout.leftMargin:    mainwindow.width*0.025//32

                        }

                        Label {
                            text: qsTr("%1 images processed.").arg(backend.totalCount)
                            textFormat: Text.RichText
                            height:  processStatusResult.visible ? 32 : 0
                            font.pixelSize:12
                            font.family: "Segoe UI"
                            horizontalAlignment :Text.AlignLeft
                            verticalAlignment: Text.AlignVCenter
                            color: "#FFFFFF"
                            Layout.topMargin: processStatusResult.visible ? 52 : 0
                            Layout.rightMargin: mainwindow.width*0.025//32


                        }
                    }
                }

                //Results
                Label {
                    id:lblResults
                    text: qsTr("Results - Search categories")
                    textFormat: Text.RichText
                    height: 20
                    font.pixelSize:18
                    font.family: "Segoe UI"
                    // horizontalAlignment :Text.AlignLeft
                    //verticalAlignment: Text.AlignVCenter
                    color: "#FFFFFF"
                    Layout.topMargin: 12
                    visible: false

                }
                Rectangle
                {
                    id: rectSearch
                    Layout.topMargin: 0
                    color: "#202020"
                    border.color: "#202020"

                    width: (mainwindow.width - (mainwindowmargin*2))
                    Layout.preferredWidth: (mainwindow.width - (mainwindowmargin*2))
                    height: 28
                    Layout.preferredHeight:28
                    radius: 4
                    visible: false

                    Row
                    {
                        anchors.fill: rectSearch
                        spacing: 16

                        ComboBox {
                            id:txtInputSearch
                            Layout.preferredHeight: 28
                            height: 28
                            editable : true

                            delegate: ItemDelegate {
                                id: cmbxDelegate
                                width: txtInputSearch.width
                                height:  28
                                padding: 0
                                hoverEnabled: true
                                spacing: 0

                                /*
                            CheckBox {
                                id: checkboxId
                                height: parent.height
                                width: height
                               // onPressed: checked = !checked
                                onCheckedChanged: {
                                    if(checked)
                                    {
                                         //model.append({text: editText})
                                       // listmodelId.append({ "name": name, "fill": fill })
                                    }
                                }
                            }

                            */
                                contentItem: Text {
                                    text: modelData
                                    color: "#FFFFFF"
                                    elide: Text.ElideRight
                                    verticalAlignment: Text.AlignVCenter
                                    leftPadding: 16 //+checkboxId.width
                                }

                                highlighted: txtInputSearch.highlightedIndex === index
                                background: Rectangle {
                                    implicitWidth: txtInputSearch.width
                                    implicitHeight: txtInputSearch.height
                                    color: txtInputSearch.highlightedIndex === index ? "#404040" : "#282828"
                                }
                            }

                            indicator: Canvas {
                                id: canvas
                                x: txtInputSearch.width - width - txtInputSearch.rightPadding
                                y: txtInputSearch.topPadding + (txtInputSearch.availableHeight - height) / 2
                                width: 12
                                height: 8
                                contextType: "2d"

                                Connections {
                                    target: txtInputSearch
                                    function onPressedChanged() { canvas.requestPaint(); }
                                }

                                onPaint: {
                                    context.reset();
                                    context.moveTo(0, 0);
                                    context.lineTo(width, 0);
                                    context.lineTo(width / 2, height);
                                    context.closePath();
                                    context.fillStyle = txtInputSearch.pressed ? "#888888" :  "#686868";
                                    context.fill();
                                }
                            }

                            contentItem: Text {
                                text: txtInputSearch.displayText
                                color: "#FFFFFF"
                                elide: Text.ElideRight
                                verticalAlignment: Text.AlignVCenter
                                leftPadding: 8
                            }

                            background: Rectangle {
                                implicitWidth: 150
                                implicitHeight: 28
                                border.color: "#404040"
                                color: "#404040"
                                border.width: 1
                                radius: 2
                            }

                            popup: Popup {
                                id:cmbxPopupMenu
                                y:28
                                width: txtInputSearch.width
                                implicitHeight: 150
                                leftPadding   : 2
                                rightPadding  : 2
                                topPadding    : 2
                                bottomPadding : 2

                                contentItem: ListView {
                                    id:listviewInputSearch
                                    clip: true
                                    implicitHeight: 28
                                    model:  txtInputSearch.delegateModel
                                    currentIndex: txtInputSearch.highlightedIndex
                                    ScrollIndicator.vertical: ScrollIndicator { }
                                    spacing:  2
                                }

                                background: Rectangle {
                                    border.color: "#303030"
                                    border.width: 1
                                    radius: 2
                                    color: "#282828"
                                }

                            }

                            model: ListModel {
                                id: searchCmbxModel
                            }
                            onAccepted: {
                                if (find(editText) === -1)
                                    model.append({text: editText})
                            }
                            Layout.preferredWidth:  parent.width - searchBtn.width -16
                            width:  parent.width - searchBtn.width -16


                        }

                        //Browse Button
                        Button {
                            id:searchBtn
                            text : qsTr("Search")
                            Layout.alignment: Qt.AlignRight
                            font.family:"Segoe UI"
                            font.pixelSize: 14
                            width: mainwindow.width*0.0609375//78
                            height:  28
                            flat: true
                            visible: true
                            Layout.topMargin: 16

                            contentItem: Text {
                                text: searchBtn.text
                                font: searchBtn.font
                                topPadding:-2
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                                color: "#FFFFFF"
                            }

                            onClicked:
                            {
                                searchCategory = txtInputSearch.displayText
                            }

                            background: Rectangle {
                                id:rectSearchBtn
                                implicitWidth:  mainwindow.width*0.0609375//78
                                implicitHeight: 28
                                width:   mainwindow.width*0.0609375//78
                                height:  28
                                border.width:1
                                radius: 1
                                border.color: "#00000040"
                                color:  searchBtn.pressed? "#228b22" : ( searchBtn.hovered ?  "#686868" : "#404040")//searchBtn.hovered ? "#686868" : "#404040"
                            }

                        }

                    }



                }


                //Results
                Label {
                    id:lblCategories
                    text: qsTr("Categories - Select category to see images")
                    textFormat: Text.RichText
                    height: 20
                    font.pixelSize:18
                    font.family: "Segoe UI"
                    // horizontalAlignment :Text.AlignLeft
                    //verticalAlignment: Text.AlignVCenter
                    color: "#FFFFFF"
                    Layout.topMargin: 12
                    visible: false

                }

                Rectangle
                {
                    id:rectHist
                    Layout.topMargin: 12
                    color: "#202020"
                    border.color: "#202020"
                    width: (mainwindow.width - (mainwindowmargin*2))
                    Layout.preferredWidth:(mainwindow.width - (mainwindowmargin*2))
                    Layout.preferredHeight: mainwindow.height*0.4
                    height:  mainwindow.height*0.4

                    radius: 4
                    visible: false

                    Rectangle {
                        color: "transparent"
                        border.color: "transparent"
                        id: rlCount
                        //spacing:0
                        anchors.topMargin:-12
                        anchors.fill:rectHist

                        Label {
                            text: "0"
                            textFormat: Text.RichText
                            height: 12
                            font.pixelSize:12
                            font.family: "Segoe UI"
                            horizontalAlignment :Text.AlignLeft
                            verticalAlignment: Text.AlignVCenter
                            color: "#FFFFFF"
                            anchors.left:parent.left
                            id:txtMinImageCount
                            visible: false

                        }

                        Label {
                            text: backend.categoryMaxBarCount
                            textFormat: Text.RichText
                            height: 12
                            font.pixelSize:12
                            font.family: "Segoe UI"
                            horizontalAlignment :Text.AlignRight
                            verticalAlignment: Text.AlignVCenter
                            color: "#FFFFFF"
                            anchors.right:parent.right
                            id:txtMaxImageCount
                            visible: false

                        }
                    }


                    Flickable {
                        id: flick
                        height: mainwindow.height*0.4
                        width: (mainwindow.width - (mainwindowmargin*2))
                        contentHeight: column.height
                        clip: true

                        ScrollBar.vertical: ScrollBar {
                            id:toolDetailsScrollbar
                            parent: flick
                            height: flick.availableHeight
                            interactive: true
                            width: 12
                            anchors.right: flick.right
                            implicitWidth:12
                            policy: ScrollBar.AsNeeded
                        }

                        Column {
                            id: column
                            leftPadding: 8
                            spacing:4
                            topPadding:8

                            Repeater {
                                model: histListModal.model
                                delegate: Component {
                                    Rectangle
                                    {
                                        id: barArea
                                        color: "transparent"
                                        border.color: "transparent"
                                        width: rectHist.width - 40
                                        height:  28

                                        visible :
                                        {
                                            if(searchCategory === "" || searchCategory === "Show All Categories" || searchCategory === histListModal.model[index].Category)
                                            {
                                                true
                                            }else
                                            {
                                                false
                                            }
                                        }

                                        Rectangle
                                        {
                                            id:rectbarPart
                                            anchors.left: parent.left
                                            anchors.top: parent.top
                                            anchors.bottom: parent.bottom
                                            width: parent.width * (histListModal.model[index].BarLength)
                                            radius: 4
                                            color: rectbarPartMouseHoverHandler.hovered || categoryTextMouseHoverHandler.hovered ? "#686868" : "#505050"
                                            border.color: "#505050"

                                            HoverHandler {
                                                id: rectbarPartMouseHoverHandler
                                                acceptedDevices: PointerDevice.Mouse
                                                cursorShape: Qt.PointingHandCursor
                                            }

                                            MouseArea {
                                                anchors.fill: parent
                                                onClicked: { backend.openFolder(histListModal.model[index].FolderPath) }
                                            }

                                            Text {
                                                id:txtCount
                                                text: histListModal.model[index].Count
                                                color: "#FFFFFF"
                                                horizontalAlignment: Qt.AlignRight
                                                verticalAlignment: Qt.AlignVCenter
                                                width: barArea.width
                                                anchors.right: parent.right
                                                visible: rectbarPart.width > categoryText.implicitWidth
                                                rightPadding:  2
                                                topPadding:8

                                            }
                                        }

                                        Text {
                                            id:categoryText
                                            text: (txtCount.visible === false) ? histListModal.model[index].Category+" "+histListModal.model[index].Count : histListModal.model[index].Category
                                            color:"#FFFFFF"
                                            horizontalAlignment: Qt.AlignLeft
                                            width : parent.width
                                            verticalAlignment: Qt.AlignVCenter
                                            leftPadding:8
                                            topPadding:8

                                            HoverHandler {
                                                id: categoryTextMouseHoverHandler
                                                acceptedDevices: PointerDevice.Mouse
                                                cursorShape: Qt.PointingHandCursor
                                            }

                                            MouseArea {
                                                anchors.fill: parent
                                                onClicked: { backend.openFolder(histListModal.model[index].FolderPath) }
                                            }
                                        }


                                    }
                                }
                            }
                        }

                    }

                }


                //Face Detection
                Label {
                    id:lblFaceDetection
                    text: qsTr("Face Detection - Select category to see images")
                    textFormat: Text.RichText
                    height: 20
                    font.pixelSize:18
                    font.family: "Segoe UI"
                    // horizontalAlignment :Text.AlignLeft
                    //verticalAlignment: Text.AlignVCenter
                    color: "#FFFFFF"
                    Layout.topMargin: 12
                    //visible: rectHist.visible
                    visible: false

                }

                Rectangle
                {
                    id:rectFaceDetectHist
                    Layout.topMargin: 12
                    color: "#202020"
                    border.color: "#202020"
                    width: (mainwindow.width - (mainwindowmargin*2))
                    Layout.preferredWidth:(mainwindow.width - (mainwindowmargin*2))
                    Layout.preferredHeight: mainwindow.height*0.4
                    height:  mainwindow.height*0.4

                    Layout.bottomMargin: 50

                    radius: 4
                    //visible: rectHist.visible
                    visible: false 
                    Rectangle {
                        color: "transparent"
                        border.color: "transparent"
                        id: rlFaceDetectCount
                        //spacing:0
                        anchors.topMargin:-12
                        anchors.fill:rectFaceDetectHist

                        Label {
                            text: "0"
                            textFormat: Text.RichText
                            height: 12
                            font.pixelSize:12
                            font.family: "Segoe UI"
                            horizontalAlignment :Text.AlignLeft
                            verticalAlignment: Text.AlignVCenter
                            color: "#FFFFFF"
                            anchors.left:parent.left
                            id:txtFaceDetectMinImageCount
                            visible: rectFaceDetectHist.visible

                        }

                        Label {
                            text: backend.faceDetectMaxBarCount
                            textFormat: Text.RichText
                            height: 12
                            font.pixelSize:12
                            font.family: "Segoe UI"
                            horizontalAlignment :Text.AlignRight
                            verticalAlignment: Text.AlignVCenter
                            color: "#FFFFFF"
                            anchors.right:parent.right
                            id:txtFaceDetectMaxImageCount
                            visible: rectFaceDetectHist.visible

                        }
                    }


                    Flickable {
                        id: flickFaceDetect
                        height: mainwindow.height*0.4
                        width: (mainwindow.width - (mainwindowmargin*2))
                        contentHeight: columnFaceDetect.height
                        clip: true

                        ScrollBar.vertical: ScrollBar {
                            id:faceDetectHistScrollbar
                            parent: flickFaceDetect
                            height: flickFaceDetect.availableHeight
                            interactive: true
                            width: 12
                            anchors.right: flickFaceDetect.right
                            implicitWidth:12
                            policy: ScrollBar.AsNeeded
                        }

                        Column {
                            id: columnFaceDetect
                            leftPadding: 8
                            spacing:4
                            topPadding:8

                            Repeater {
                                model: histListModalFaceDetect.model
                                delegate: Component {
                                    Rectangle
                                    {
                                        id: barAreaFaceDetect
                                        color: "transparent"
                                        border.color: "transparent"
                                        width: rectHist.width - 40
                                        height:  28

                                        Rectangle
                                        {
                                            id:rectbarPartFaceDetect
                                            anchors.left: parent.left
                                            anchors.top: parent.top
                                            anchors.bottom: parent.bottom
                                            width: parent.width * (histListModalFaceDetect.model[index].BarLength)
                                            radius: 4
                                            color: rectbarFaceDetectPartMouseHoverHandler.hovered || faceDetectCategoryTextMouseHoverHandler.hovered ? "#686868" : "#505050"
                                            border.color: "#505050"

                                            HoverHandler {
                                                id: rectbarFaceDetectPartMouseHoverHandler
                                                acceptedDevices: PointerDevice.Mouse
                                                cursorShape: Qt.PointingHandCursor
                                            }

                                            MouseArea {
                                                anchors.fill: parent
                                                onClicked: { backend.openFolder(histListModalFaceDetect.model[index].FolderPath) }
                                            }

                                            Text {
                                                id:txtFaceDetectCount
                                                text: histListModalFaceDetect.model[index].Count
                                                color: "#FFFFFF"
                                                horizontalAlignment: Qt.AlignRight
                                                verticalAlignment: Qt.AlignVCenter
                                                width: barAreaFaceDetect.width
                                                anchors.right: parent.right
                                                visible: rectbarPartFaceDetect.width > faceDetectCategoryText.implicitWidth
                                                rightPadding:  2
                                                topPadding:8

                                            }
                                        }

                                        Text {
                                            id:faceDetectCategoryText
                                            text: (txtFaceDetectCount.visible === false) ? histListModalFaceDetect.model[index].Category+"    "+histListModalFaceDetect.model[index].Count : histListModalFaceDetect.model[index].Category
                                            color:"#FFFFFF"
                                            horizontalAlignment: Qt.AlignLeft
                                            width : parent.width
                                            verticalAlignment: Qt.AlignVCenter
                                            leftPadding:8
                                            topPadding:8

                                            HoverHandler {
                                                id: faceDetectCategoryTextMouseHoverHandler
                                                acceptedDevices: PointerDevice.Mouse
                                                cursorShape: Qt.PointingHandCursor
                                            }

                                            MouseArea {
                                                anchors.fill: parent
                                                onClicked: { backend.openFolder(histListModalFaceDetect.model[index].FolderPath) }
                                            }
                                        }


                                    }
                                }
                            }
                        }

                    }

                }


            }
        } // content area flickable

    }



    Connections {
        target: backend

        function onScanProcessingCompleted()
        {
            searchCmbxModel.clear()
            searchCmbxModel.append({"text":"Show All Categories"})
            for (var nCount = 0; nCount < histListModal.model.length; nCount++)  {
                searchCmbxModel.append({"text": histListModal.model[nCount].Category})
            }

            busyIndication.visible = false
            processingProgress.visible = false

            startScanWithoutIPUBtnText.text = "Scan with Azure EP"
            startScanBtnText.text = "Scan with Ryzen AI"

            txtInputLocationPath.visible = true
            browseBtn.visible = true
            startScanBtn.visible = true
            scanWithoutIPUBtn.visible = true
            processStatusResult.visible =true

            lblResults.visible = true
            rectSearch.visible = true

            lblCategories.visible = true
            txtMaxImageCount.visible = true
            txtMinImageCount.visible = true
            rectHist.visible = true

        }

        function onScanWithoutIPUProcessingCompleted()
        {

            searchCmbxModel.clear()
            searchCmbxModel.append({"text":"Show All Categories"})
            for (var nCount = 0; nCount < histListModal.model.length; nCount++)  {
                searchCmbxModel.append({"text": histListModal.model[nCount].Category})
            }

            busyIndication.visible = false
            processingProgress.visible = false

            startScanWithoutIPUBtnText.text = "Scan with Azure EP"
            startScanBtnText.text = "Scan with Ryzen AI"

            txtInputLocationPath.visible = true
            browseBtn.visible = true
            startScanBtn.visible = true
            scanWithoutIPUBtn.visible = true
            processStatusResult.visible =true

            lblResults.visible = true
            rectSearch.visible = true

            lblCategories.visible = true
            txtMaxImageCount.visible = true
            txtMinImageCount.visible = true
            rectHist.visible = true



        }

        function onShowWarningMessage(){

            busyIndication.visible = false
            processingProgress.visible = false

            startScanWithoutIPUBtnText.text = "Scan with Azure EP"
            startScanBtnText.text = "Scan with Ryzen AI"

            txtInputLocationPath.visible = true
            browseBtn.visible = true
            startScanBtn.visible = true
            scanWithoutIPUBtn.visible = true
            processStatusResult.visible =true

            lblCategories.visible = false
            txtMaxImageCount.visible = false
            txtMinImageCount.visible = false
            lblResults.visible = false
            rectSearch.visible = false
            rectHist.visible = false

            msgDialogText = backend.previousExceptionMessage
            msgDialog.visible = true
        }


    }

}
